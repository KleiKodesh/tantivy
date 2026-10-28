//! Best-effort lowering of the OS scheduling priority of the calling thread,
//! used by `IndexWriterOptions::low_priority_threads`.

/// Lowers the scheduling priority of the calling thread to below normal.
///
/// Failures are ignored: priority is a hint, indexing works either way.
pub(crate) fn lower_current_thread_priority() {
    imp::lower();
}

#[cfg(windows)]
mod imp {
    use winapi::um::processthreadsapi::{GetCurrentThread, SetThreadPriority};
    use winapi::um::winbase::THREAD_PRIORITY_BELOW_NORMAL;

    pub(super) fn lower() {
        // SAFETY: `GetCurrentThread` returns a pseudo handle valid for the
        // calling thread; `SetThreadPriority` only reads it.
        unsafe {
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL as i32);
        }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    /// On Linux, `setpriority(PRIO_PROCESS, tid, ..)` sets the nice value of a
    /// single thread.
    pub(super) fn lower() {
        // SAFETY: plain syscalls on the calling thread's own id.
        unsafe {
            let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
            let current = libc::getpriority(libc::PRIO_PROCESS, tid);
            let target = (current + 10).min(19);
            libc::setpriority(libc::PRIO_PROCESS, tid, target);
        }
    }
}

#[cfg(target_vendor = "apple")]
mod imp {
    /// On Apple platforms, the utility QoS class lowers both CPU priority and
    /// I/O throttling of the calling thread.
    pub(super) fn lower() {
        // SAFETY: only affects the calling thread.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0);
        }
    }
}

#[cfg(not(any(windows, target_os = "linux", target_vendor = "apple")))]
mod imp {
    pub(super) fn lower() {}
}

#[cfg(test)]
mod tests {
    #[test]
    fn lowering_the_priority_does_not_panic() {
        std::thread::spawn(super::lower_current_thread_priority)
            .join()
            .unwrap();
    }
}
