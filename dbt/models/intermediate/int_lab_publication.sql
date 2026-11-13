-- Papers credited to a lab's grants (NIH link tables), one row per lab x paper.
select distinct la.inv_id, p.pub_id, p.pmid, p.title, p.journal, p.pub_year
from {{ ref('int_lab_award') }} la
join {{ ref('stg_publication') }} p using (award_id)
