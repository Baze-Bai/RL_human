-- Run this in the Supabase SQL Editor to create the preferences table.

create table if not exists preferences (
    id          uuid primary key,
    created_at  timestamptz not null default now(),
    prompt      text not null,
    response_a  text not null,
    response_b  text not null,
    chosen      text not null,
    rejected    text not null,
    preference  text not null check (preference in ('response_a', 'response_b', 'tie')),
    model       text,
    temperature real,
    meta_a      jsonb,
    meta_b      jsonb
);

-- Row-level security: allow inserts and reads from the anon key
alter table preferences enable row level security;

create policy "Allow anonymous inserts"
    on preferences for insert
    with check (true);

create policy "Allow anonymous reads"
    on preferences for select
    using (true);
