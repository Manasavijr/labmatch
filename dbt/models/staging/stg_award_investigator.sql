select award_id, inv_id, role from {{ source('bronze', 'award_investigator') }}
