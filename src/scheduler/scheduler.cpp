#include "scheduler.h"
#include "core/execution_plan.h"
#include "core/engine.h"
#include "model/quantization_utils.h"

#include <iostream>
#include <vector>

namespace engine {

Scheduler::Scheduler() = default;

uint64_t Scheduler::submit(const core::ExecutionPlan& plan, int priority) {
    Job job;
    job.id = next_id_++;
    job.plan = plan;
    job.priority = priority;
    job.status = JobStatus::Pending;

    queue_.push(job);

    std::cerr
        << "[scheduler] job submitted id=" << job.id
        << " priority=" << job.priority << "\n";

    return job.id;
}

bool Scheduler::run_next() {
    if (queue_.empty()) {
        return false;
    }

    Job job = queue_.top();
    queue_.pop();

    job.status = JobStatus::Running;

    std::cerr
        << "[scheduler] running job id=" << job.id
        << " priority=" << job.priority << "\n";

    Engine engine;
    engine.run("model.gguf", job.plan);

    job.status = JobStatus::Finished;
    std::cerr << "[scheduler] job finished id=" << job.id << "\n";

    return true;
}

bool Scheduler::compatible(const Job& a, const Job& b) const {
    return a.plan.quantization == b.plan.quantization;
}

size_t Scheduler::run_batch() {
    if (queue_.empty()) {
        return 0;
    }

    Job first = queue_.top();
    queue_.pop();

    std::vector<Job> batch;
    batch.push_back(first);

    while (!queue_.empty()) {
        const Job& next = queue_.top();
        if (!compatible(first, next)) {
            break;
        }
        batch.push_back(next);
        queue_.pop();
    }

    std::cerr
        << "[scheduler] running batch size=" << batch.size()
        << " quant=" << quant_to_string(first.plan.quantization)
        << "\n";

    for (auto& job : batch) {
        job.status = JobStatus::Running;

        std::cerr
            << "[scheduler] running job id=" << job.id
            << " priority=" << job.priority << "\n";

        Engine engine;
        engine.run("model.gguf", job.plan);

        job.status = JobStatus::Finished;
        std::cerr << "[scheduler] job finished id=" << job.id << "\n";
    }

    return batch.size();
}

bool Scheduler::empty() const {
    return queue_.empty();
}

} // namespace engine