# ADR-0001: Cell-based, multi-region active-active

- **Status:** Deferred — not needed at current load (6k–10k/day); see architecture §0 for the trigger to adopt
- **Date:** 2026-09-26

## Context
We need to serve 1M+ concurrent users worldwide with spiky load, 99.95% availability, and no dependence on a single cloud. One large cluster per region would put every user in the same failure domain for a bad deploy, a noisy tenant, or a saturated dependency such as Redis.

## Decision
- Run **active-active in at least 3 regions**. Global edge geo-steering sends users to their nearest healthy region.
- Split each region into **cells**. A cell is a complete, independent copy of the stack: gateway, APIs, AI Gateway, agent workers, Redis, and queues. Each cell has a fixed capacity ceiling (starting point: 100k concurrent users).
- A thin routing layer maps each tenant/user to a home cell with consistent hashing. The mapping is stored in the global control-plane DB.
- The distributed SQL database spans cells within a region. Data is homed per user region for residency.
- Deploys go cell by cell (canary cell first) with automatic halt on SLO regression.

## Consequences
- ✅ Blast radius is limited to one cell. Scaling means adding cells, and each cell is load-tested once.
- ✅ Large tenants can get dedicated cells.
- ❌ More operational overhead: many clusters or namespaces to manage. This needs strong IaC and GitOps from day 0.
- ❌ Cross-cell features (global search, analytics) must go through the event bus or data warehouse, not direct queries.
