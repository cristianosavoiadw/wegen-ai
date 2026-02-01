#pragma once

#include <queue>
#include <vector>
#include <atomic>

#include "core/execution_plan.h"

namespace core {
    struct ExecutionPlan;
}

namespace engine {

enum class JobStatus {
    Pending,
    Running,
    Finished,
    Failed
};

struct Job {
    uint64_t id;
    int priority = 0;
    core::ExecutionPlan plan;
    JobStatus status = JobStatus::Pending;
    int exit_code = -1;
};

struct JobCompare {
    bool operator()(const Job& a, const Job& b) const {
        if (a.priority == b.priority) {
            return a.id > b.id;
        }
        return a.priority < b.priority;
    }
};

class Scheduler {
public:
    Scheduler();

    uint64_t submit(const core::ExecutionPlan& plan, int priority = 0);

    bool run_next();
    size_t run_batch();
    bool empty() const;

private:
    std::priority_queue<Job, std::vector<Job>, JobCompare> queue_;
    std::atomic<uint64_t> next_id_{1};

    bool compatible(const Job& a, const Job& b) const;
};

} // namespace engine