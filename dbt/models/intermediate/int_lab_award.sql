-- Every award held by a PI who has at least one AI/ML award at a US institution.
with ml_pis as (
    select distinct ai.inv_id
    from {{ ref('stg_award_investigator') }} ai
    join {{ ref('stg_award') }} a using (award_id)
    join {{ ref('stg_institution') }} i on i.inst_id = a.inst_id
    where a.is_ml and i.is_us
)
select ai.inv_id, ai.role, a.*
from {{ ref('stg_award_investigator') }} ai
join ml_pis using (inv_id)
join {{ ref('stg_award') }} a using (award_id)
