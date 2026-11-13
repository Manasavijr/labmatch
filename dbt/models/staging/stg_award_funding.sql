select award_id, fiscal_year, amount from {{ source('bronze', 'award_funding') }}
