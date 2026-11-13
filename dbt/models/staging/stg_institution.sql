select inst_id, name as inst_name, city, state, country,
       upper(coalesce(country, '')) in ('UNITED STATES', 'USA', 'US') as is_us
from {{ source('bronze', 'institution') }}
