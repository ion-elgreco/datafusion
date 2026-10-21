// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`FileList`]: files of a scan that are listed while the scan runs.

use std::fmt::Debug;
use std::sync::Arc;

use datafusion_common::Result;
use datafusion_execution::TaskContext;
use futures::stream::BoxStream;

use crate::PartitionedFile;

/// A stream of files to scan, see [`FileList`].
pub type PartitionedFileStream = BoxStream<'static, Result<PartitionedFile>>;

/// Lists the files of a file scan while the scan runs.
///
/// Usually all files of a scan are known at planning time, and are set in
/// [`FileScanConfig::file_groups`]. With a `FileList`, a scan can also read
/// files that are only known during execution. For example, a table format
/// can replay its log lazily and send each file to the scan when the replay
/// finds it, or it can skip files with information that exists only at
/// runtime.
///
/// A scan opens its file list once per execution, when one of its partitions
/// needs a file and the planned files are used up. All partitions of the scan
/// take the listed files from one shared queue, so each file is read once. The
/// scan stops polling the stream while that queue is full.
///
/// The files are not known at planning time, so a scan with a file list has
/// unknown statistics, and it must not declare an output ordering.
///
/// [`FileScanConfig::file_groups`]: crate::file_scan_config::FileScanConfig::file_groups
pub trait FileList: Debug + Send + Sync {
    /// Open the stream of files for one execution of the scan.
    fn open(&self, context: Arc<TaskContext>) -> Result<PartitionedFileStream>;
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{AsArray, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema, SchemaRef};
    use datafusion_common::{exec_err, stats::Precision};
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_physical_plan::{ExecutionPlan, common::collect};
    use futures::{FutureExt, StreamExt, future::try_join_all};

    use super::*;
    use crate::TableSchema;
    use crate::file_groups::FileGroup;
    use crate::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
    use crate::file_stream::{FileOpenFuture, FileOpener};
    use crate::source::{DataSource, DataSourceExec};
    use crate::test_util::MockSource;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "file",
            DataType::Int32,
            false,
        )]))
    }

    /// Returns one row per file: the number in the name of the file.
    struct FileNumberOpener;

    impl FileOpener for FileNumberOpener {
        fn open(&self, file: PartitionedFile) -> Result<FileOpenFuture> {
            let number: i32 = file
                .path()
                .as_ref()
                .trim_start_matches("file")
                .parse()
                .unwrap();
            let batch = RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int32Array::from(vec![number]))],
            )?;
            let stream = futures::stream::iter([Ok(batch)]).boxed();
            Ok(futures::future::ready(Ok(stream)).boxed())
        }
    }

    /// Lists `file0` to `file{count - 1}`, and yields to the runtime before each file.
    #[derive(Debug, Default)]
    struct TestFileList {
        count: i32,
        /// Fail instead of listing this file
        fail_at: Option<i32>,
        opened: AtomicUsize,
    }

    impl FileList for TestFileList {
        fn open(&self, _context: Arc<TaskContext>) -> Result<PartitionedFileStream> {
            self.opened.fetch_add(1, Ordering::SeqCst);
            let fail_at = self.fail_at;
            Ok(futures::stream::iter(0..self.count)
                .then(move |number| async move {
                    tokio::task::yield_now().await;
                    if Some(number) == fail_at {
                        return exec_err!("cannot list file{number}");
                    }
                    Ok(PartitionedFile::new(format!("file{number}"), 10))
                })
                .boxed())
        }
    }

    fn scan(
        planned: Vec<&str>,
        partitions: usize,
        file_list: Arc<TestFileList>,
    ) -> Arc<DataSourceExec> {
        let source = MockSource::new(TableSchema::from(schema()))
            .with_file_opener(Arc::new(FileNumberOpener));
        let mut file_groups = vec![FileGroup::new(vec![]); partitions];
        file_groups[0] = FileGroup::new(
            planned
                .into_iter()
                .map(|path| PartitionedFile::new(path, 10))
                .collect(),
        );
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(source),
        )
        .with_file_groups(file_groups)
        .with_file_list(file_list)
        .build();
        DataSourceExec::from_data_source(config)
    }

    /// Execute all partitions of `plan` at the same time. Returns the file numbers sorted.
    async fn read_all(plan: &Arc<DataSourceExec>) -> Result<Vec<i32>> {
        let context = Arc::new(TaskContext::default());
        let partitions = plan.properties().partitioning.partition_count();
        let streams = (0..partitions)
            .map(|partition| plan.execute(partition, Arc::clone(&context)))
            .collect::<Result<Vec<_>>>()?;
        let batches = try_join_all(streams.into_iter().map(collect)).await?;
        let mut numbers: Vec<i32> = batches
            .iter()
            .flatten()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        numbers.sort_unstable();
        Ok(numbers)
    }

    #[tokio::test]
    async fn file_list_reads_each_file_once() -> Result<()> {
        let file_list = Arc::new(TestFileList {
            count: 200,
            ..Default::default()
        });
        let plan = scan(vec![], 4, Arc::clone(&file_list));

        assert_eq!(read_all(&plan).await?, (0..200).collect::<Vec<_>>());
        assert_eq!(file_list.opened.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn file_list_follows_planned_files() -> Result<()> {
        let file_list = Arc::new(TestFileList {
            count: 3,
            ..Default::default()
        });
        let plan = scan(vec!["file100", "file101"], 2, file_list);

        assert_eq!(read_all(&plan).await?, vec![0, 1, 2, 100, 101]);
        Ok(())
    }

    #[tokio::test]
    async fn file_list_is_opened_again_after_reset() -> Result<()> {
        let file_list = Arc::new(TestFileList {
            count: 5,
            ..Default::default()
        });
        let plan = scan(vec![], 2, Arc::clone(&file_list));
        assert_eq!(read_all(&plan).await?, vec![0, 1, 2, 3, 4]);

        let plan = Arc::clone(&plan).reset_state()?;
        let plan = Arc::new(plan.downcast_ref::<DataSourceExec>().unwrap().clone());
        assert_eq!(read_all(&plan).await?, vec![0, 1, 2, 3, 4]);
        assert_eq!(file_list.opened.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn file_list_error_fails_the_scan() -> Result<()> {
        let file_list = Arc::new(TestFileList {
            count: 10,
            fail_at: Some(5),
            ..Default::default()
        });
        let plan = scan(vec![], 3, file_list);

        let err = read_all(&plan).await.unwrap_err();
        assert!(err.to_string().contains("cannot list file5"), "{err}");
        Ok(())
    }

    #[test]
    fn file_list_has_unknown_statistics_and_can_repartition() -> Result<()> {
        let plan = scan(vec!["file7"], 1, Arc::default());
        let config = plan.data_source().downcast_ref::<FileScanConfig>().unwrap();
        let statistics = config.partition_statistics(None)?;
        assert_eq!(statistics.num_rows, Precision::Absent);

        let repartitioned = config.repartitioned(4, 0, None)?.unwrap();
        assert_eq!(repartitioned.output_partitioning().partition_count(), 4);
        assert!(config.repartitioned(1, 0, None)?.is_none());
        Ok(())
    }
}
