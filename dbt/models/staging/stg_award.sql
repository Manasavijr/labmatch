select
    award_id,
    agency,
    inst_id,
    source_award_no,
    title,
    abstract,
    program,
    start_date,
    end_date,
    total_amount,
    end_date >= date '{{ var("as_of_date") }}' as is_active,
    -- training, travel and event awards are not lab research funding
    not regexp_matches(lower(coalesce(title, '')), '{{ var("non_research_pattern") }}') as is_research,
    regexp_matches(lower(coalesce(title, '') || ' ' || coalesce(abstract, '')), '{{ var("ml_pattern") }}') as is_ml
from {{ source('bronze', 'award') }}
