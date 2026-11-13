-- Text units for retrieval: each grant abstract and each grant-linked paper title, tagged with its lab.
select inv_id, 'grant' as doc_type, cast(award_id as varchar) as doc_id,
       coalesce(title, '') || '. ' || coalesce(abstract, '') as text,
       year(start_date) as year
from {{ ref('int_lab_award') }}
union all
select inv_id, 'paper', cast(pub_id as varchar), title, pub_year
from {{ ref('int_lab_publication') }}
where title is not null
