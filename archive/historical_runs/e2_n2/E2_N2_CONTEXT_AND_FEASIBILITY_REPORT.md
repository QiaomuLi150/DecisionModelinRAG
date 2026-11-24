# E2-N2 Context and Feasibility Report

Status: `PASS_WITH_LIMITATIONS`

All locally cached checkpoints were available for the authenticated source runs. Native input feasibility was measured in those traces and preserved in the raw rows. Intern-Decision has 500 enterprise rows blocked by its native 8192-token contract; those rows remain explicit failures and are not silently truncated or removed.

The complete historical workload is 3,767 Track A questions plus 500 Track B enterprise questions, with 20 candidates per query. Track A and Track B remain separate. The E2-N2 scorecard reports completed-inference and context-feasible denominators separately from the full workload.

The common-information quality metrics are complete-support selection proxies, not reader factual accuracy. Native optimized deployment and cross-model denominator differences must not be treated as a fair common-context comparison without qualification.
