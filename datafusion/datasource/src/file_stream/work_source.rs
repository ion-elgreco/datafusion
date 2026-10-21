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

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll, Waker};

use crate::PartitionedFile;
use crate::file_groups::FileGroup;
use crate::file_list::{FileList, PartitionedFileStream};
use crate::file_scan_config::FileScanConfig;
use datafusion_common::{DataFusionError, Result, internal_datafusion_err};
use datafusion_common_runtime::SpawnedTask;
use datafusion_execution::TaskContext;
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::sync::Notify;

/// Number of listed files that can wait in the queue. The scan stops polling
/// its [`FileList`] while the queue is full.
const LISTED_FILES_CAPACITY: usize = 64;

/// Source of work for `ScanState`.
///
/// Streams that may share work across siblings use [`WorkSource::Shared`],
/// while streams that can not share work (e.g. because they must preserve file
/// order) use  [`WorkSource::Local`].
#[derive(Debug, Clone)]
pub(super) enum WorkSource {
    /// Files this stream will plan locally without sharing them.
    Local(VecDeque<PartitionedFile>),
    /// Files shared with sibling streams.
    Shared(SharedWorkSource),
}

impl WorkSource {
    /// Poll the next file to plan from this work source.
    ///
    /// Returns `Ready(Ok(None))` when no file is left. Returns `Pending` while
    /// a [`FileList`] has not listed the next file yet.
    pub(super) fn poll_next_file(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Option<PartitionedFile>>> {
        match self {
            Self::Local(files) => Poll::Ready(Ok(files.pop_front())),
            Self::Shared(shared) => shared.poll_next_file(cx),
        }
    }

    /// Return how many queued files should be counted as already processed
    /// when this stream stops early after hitting a global limit.
    pub(super) fn skipped_on_limit(&self) -> usize {
        match self {
            Self::Local(files) => files.len(),
            Self::Shared(_) => 0,
        }
    }
}

/// Shared source of work for sibling `FileStream`s
///
/// The queue is created once per execution and shared by all reorderable
/// sibling streams for that execution. Whichever stream becomes idle first may
/// take the next unopened file from the front of the queue.
///
/// When the scan has a [`FileList`], the streams take the listed files after
/// the planned files.
///
/// It uses a [`Mutex`] internally to provide thread-safe access
/// to the shared file queue.
#[derive(Debug, Clone)]
pub(crate) struct SharedWorkSource {
    inner: Arc<SharedWorkSourceInner>,
}

#[derive(Debug, Default)]
pub(super) struct SharedWorkSourceInner {
    files: Mutex<VecDeque<PartitionedFile>>,
    /// Files listed while the scan runs, taken after `files`
    listed: Option<ListedFiles>,
}

impl SharedWorkSource {
    /// Create a shared work source for the unopened files in `config`.
    ///
    /// Files are reordered by the file source (e.g. by statistics for TopK)
    /// before being placed in the shared queue, so the most promising files
    /// are processed first across all partitions. The files of the
    /// [`FileList`] of `config` follow, in the order that it lists them.
    pub(crate) fn from_config(config: &FileScanConfig) -> Self {
        let files: Vec<_> = config
            .file_groups
            .iter()
            .flat_map(FileGroup::iter)
            .cloned()
            .collect();
        let files = config.file_source.reorder_files(files);
        Self {
            inner: Arc::new(SharedWorkSourceInner {
                files: Mutex::new(files.into()),
                listed: config.file_list.clone().map(ListedFiles::new),
            }),
        }
    }

    /// Set the task context that opens the [`FileList`]. The first call sets
    /// it, later calls have no effect.
    pub(crate) fn set_task_context(&self, context: &Arc<TaskContext>) {
        if let Some(listed) = &self.inner.listed {
            listed.context.get_or_init(|| Arc::clone(context));
        }
    }

    /// Poll the next file from the shared work queue.
    ///
    /// Returns `Ready(Ok(None))` if the queue is empty and the file list, if
    /// any, has ended.
    fn poll_next_file(
        &self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Option<PartitionedFile>>> {
        if let Some(file) = self.inner.files.lock().pop_front() {
            return Poll::Ready(Ok(Some(file)));
        }
        match &self.inner.listed {
            Some(listed) => listed.poll_next_file(cx),
            None => Poll::Ready(Ok(None)),
        }
    }
}

/// The files of a [`FileList`], for one execution of the scan.
///
/// A task moves the listed files into a bounded queue, and the sibling streams
/// take them from there. The first stream that needs a listed file starts the
/// task. Dropping this value aborts the task.
struct ListedFiles {
    file_list: Arc<dyn FileList>,
    /// Opens the file list, set by the streams before they poll
    context: OnceLock<Arc<TaskContext>>,
    queue: Arc<ListedFileQueue>,
    /// The task that lists the files, or the error of opening the file list
    task: OnceLock<std::result::Result<SpawnedTask<()>, Arc<DataFusionError>>>,
}

impl fmt::Debug for ListedFiles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ListedFiles")
            .field("file_list", &self.file_list)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct ListedFileQueue {
    state: Mutex<ListedFileQueueState>,
    /// Wakes the listing task when a stream takes a file from the queue
    space: Notify,
}

#[derive(Default)]
struct ListedFileQueueState {
    files: VecDeque<PartitionedFile>,
    /// Set when the file list has ended or failed
    end: Option<std::result::Result<(), Arc<DataFusionError>>>,
    /// Streams that wait for a file
    waiters: Vec<Waker>,
}

impl ListedFiles {
    fn new(file_list: Arc<dyn FileList>) -> Self {
        Self {
            file_list,
            context: OnceLock::new(),
            queue: Arc::default(),
            task: OnceLock::new(),
        }
    }

    fn poll_next_file(
        &self,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Option<PartitionedFile>>> {
        if let Err(err) = self.task.get_or_init(|| self.start().map_err(Arc::new)) {
            return Poll::Ready(Err(DataFusionError::Shared(Arc::clone(err))));
        }

        let mut state = self.queue.state.lock();
        if let Some(file) = state.files.pop_front() {
            drop(state);
            self.queue.space.notify_one();
            return Poll::Ready(Ok(Some(file)));
        }
        match &state.end {
            Some(Ok(())) => Poll::Ready(Ok(None)),
            Some(Err(err)) => Poll::Ready(Err(DataFusionError::Shared(Arc::clone(err)))),
            None => {
                if !state.waiters.iter().any(|w| w.will_wake(cx.waker())) {
                    state.waiters.push(cx.waker().clone());
                }
                Poll::Pending
            }
        }
    }

    /// Open the file list, and start the task that moves its files into the
    /// queue.
    fn start(&self) -> Result<SpawnedTask<()>> {
        let context = self.context.get().ok_or_else(|| {
            internal_datafusion_err!("FileList opened without a task context")
        })?;
        let files = self.file_list.open(Arc::clone(context))?;
        Ok(SpawnedTask::spawn(list_files(
            files,
            Arc::clone(&self.queue),
        )))
    }
}

/// Move the files of `files` into `queue`. Wait while the queue is full.
async fn list_files(mut files: PartitionedFileStream, queue: Arc<ListedFileQueue>) {
    loop {
        loop {
            let full = queue.state.lock().files.len() >= LISTED_FILES_CAPACITY;
            if !full {
                break;
            }
            queue.space.notified().await;
        }

        let next = files.next().await;
        let mut state = queue.state.lock();
        match next {
            Some(Ok(file)) => {
                state.files.push_back(file);
                // One file needs one stream. A woken stream that finds the queue
                // empty waits again.
                if let Some(waiter) = state.waiters.pop() {
                    waiter.wake();
                }
            }
            Some(Err(err)) => state.end = Some(Err(Arc::new(err))),
            None => state.end = Some(Ok(())),
        }
        if state.end.is_some() {
            state.waiters.drain(..).for_each(Waker::wake);
            return;
        }
    }
}
