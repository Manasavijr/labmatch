"""Load the FundedLabs NIH/NSF cleaned extracts into Iceberg bronze tables.

Input: data/raw/*.tsv.gz (award, award_funding, award_investigator, award_publication,
institution, investigator, publication). Output: lake.bronze.<same names>.
"""
from pyspark.sql import functions as F, types as T
from pipeline.spark.session import spark, ROOT

S = T.StructType
f = lambda n, t=T.StringType(): T.StructField(n, t, True)
I, L, D = T.IntegerType(), T.LongType(), T.DoubleType()
SCHEMAS = {
    "award": S([f("award_id", L), f("agency_id", I), f("inst_id", L), f("source_award_no"), f("title"), f("abstract"),
                f("department"), f("program"), f("start_date"), f("end_date"), f("total_amount", D)]),
    "award_funding": S([f("award_id", L), f("fiscal_year", I), f("amount", D)]),
    "award_investigator": S([f("award_id", L), f("inv_id", L), f("role")]),
    "award_publication": S([f("award_id", L), f("pub_id", L)]),
    "institution": S([f("inst_id", L), f("name"), f("city"), f("state"), f("country")]),
    "investigator": S([f("inv_id", L), f("full_name"), f("name_norm"), f("nih_profile_id", L), f("nsf_id"), f("inst_id", L)]),
    "publication": S([f("pub_id", L), f("pmid", L), f("doi"), f("title"), f("journal"), f("pub_year", I)]),
}


def main():
    s = spark("bronze_grants")
    s.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze")
    for name, schema in SCHEMAS.items():
        df = (s.read.option("sep", "\t").option("quote", "\u0000").option("nullValue", "\\N")
              .option("mode", "PERMISSIVE").schema(schema).csv(f"{ROOT}/data/raw/{name}.tsv.gz"))
        if name == "award":
            df = (df.withColumn("start_date", F.to_date("start_date")).withColumn("end_date", F.to_date("end_date"))
                    .withColumn("agency", F.when(F.col("agency_id") == 1, "NIH").otherwise("NSF")))
        df = df.withColumn("_loaded_at", F.current_timestamp())
        df.writeTo(f"lake.bronze.{name}").using("iceberg").createOrReplace()
        print(name, s.table(f"lake.bronze.{name}").count())
    s.stop()


if __name__ == "__main__":
    main()
