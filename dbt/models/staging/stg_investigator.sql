select inv_id, full_name, name_norm,
       split_part(name_norm, '|', 1) as last_name,
       split_part(name_norm, '|', 2) as first_name,
       nih_profile_id, nsf_id, inst_id
from {{ source('bronze', 'investigator') }}
