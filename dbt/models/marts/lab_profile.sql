-- One row per lab (PI): identity, funding horizon and output features used by retrieval and the ranker.
with aw as (
    select inv_id,
           count(*) as n_awards,
           count(*) filter (where is_ml) as n_ml_awards,
           count(*) filter (where is_active and is_research) as n_active,
           coalesce(sum(total_amount) filter (where is_active and is_research), 0) as active_amount,
           max(end_date) filter (where is_active and is_research) as active_end_max,
           min(start_date) as first_start,
           max(start_date) as last_start,
           string_agg(distinct agency, '/' order by agency) as agencies
    from {{ ref('int_lab_award') }}
    group by 1
),
pubs as (
    select inv_id,
           count(*) as n_pubs,
           count(*) filter (where pub_year >= 2023) as n_pubs_recent
    from {{ ref('int_lab_publication') }}
    group by 1
)
select inv.inv_id, inv.full_name, inv.nih_profile_id, inv.nsf_id,
       ins.inst_id, ins.inst_name, ins.city, ins.state,
       aw.* exclude (inv_id),
       coalesce(pubs.n_pubs, 0) as n_pubs,
       coalesce(pubs.n_pubs_recent, 0) as n_pubs_recent,
       date_diff('month', date '{{ var("as_of_date") }}', aw.active_end_max) as months_funding_left
from aw
join {{ ref('stg_investigator') }} inv using (inv_id)
left join {{ ref('stg_institution') }} ins on ins.inst_id = inv.inst_id
left join pubs using (inv_id)
