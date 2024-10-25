# PostgreSQL

## Database Finetuning

### PgSQL Settings
- Memory
    - shared_buffers
    - work_mem
    - maintenance_work_mem
- Checkpoints
    - checkpoint_timeout
    - checkpoint_completion_target
- WAL
    - wal_buffers
    - max_wal_size

### Indexing Optimization
- Index for columns used in JOIN, WHERE, ORDER BY, GROUP BY

### Query Optimization
- Use EXPLAIN and EXPLAIN ANALYZE to check queries

### Maintenance
- VACUUM regularly

### Performance Monitoring
- pg_stat_activity
- pg_stat_statements
