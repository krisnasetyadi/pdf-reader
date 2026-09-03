-- migrations/003_skills.sql
-- Uploaded skill instructions (MS-251).
--
-- Mirrors the DDL in storage.ensure_schema(), which is what actually runs at
-- startup — this file is the readable record, like 001 and 002.
--
-- No skill_assignments table on purpose: a "team" skill belongs to the admin
-- who uploaded it and is visible to every account that admin created, which
-- users.created_by already records.

CREATE TABLE IF NOT EXISTS skills (
    id            BIGSERIAL   PRIMARY KEY,
    skill_id      TEXT        NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    name          TEXT        NOT NULL,
    slash_command TEXT        NOT NULL,
    description   TEXT        NOT NULL DEFAULT '',
    instruction   TEXT        NOT NULL,
    scope         TEXT        NOT NULL DEFAULT 'personal'
                  CHECK (scope IN ('personal', 'team')),
    owner_id      TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- owner_id leads: both visibility branches filter it by equality, and scope
-- only has two values so it barely narrows anything on its own.
CREATE INDEX IF NOT EXISTS idx_skills_owner_scope ON skills (owner_id, scope);
