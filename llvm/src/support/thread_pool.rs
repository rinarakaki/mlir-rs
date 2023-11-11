/*!
# A ThreadPool implementation

This file defines a crude C++11 based thread pool.

- include <https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Support/ThreadPool.h>
- lib <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Support/ThreadPool.cpp>
*/

/**
A ThreadPool for asynchronous parallel execution on a defined number of threads.

The pool keeps a vector of threads alive, waiting on a condition variable for some work to become available.

It is possible to reuse one thread pool for different groups of tasks by grouping tasks using ThreadPoolTaskGroup. All tasks are processed using the same queue, but it is possible to wait only for a specific group of tasks to finish.

It is also possible for worker threads to submit new tasks and wait for them. Note that this may result in a deadlock in cases such as when a task (directly or indirectly) tries to wait for its own completion, or when all available threads are used up by tasks waiting for a task that has no thread left to run on (this includes waiting on the returned future). It should be generally safe to wait() for a group as long as groups do not form a cycle.
*/
pub struct ThreadPool {
}
