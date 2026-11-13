select p.pub_id, p.pmid, p.doi, p.title, p.journal, p.pub_year, ap.award_id
from {{ source('bronze', 'publication') }} p
join {{ source('bronze', 'award_publication') }} ap using (pub_id)
