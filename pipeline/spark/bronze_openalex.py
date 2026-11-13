"""Parse OpenAlex landing files into Iceberg.

lake.bronze.openalex_pi_link : one row per (grant PI, candidate OpenAlex author) with vote count
lake.silver.pi_author        : the resolved link per PI (top-voted author, >= 2 votes or the only paper)
lake.bronze.openalex_work    : works for linked authors, abstract rebuilt from the inverted index
"""
import os
from pyspark.sql import functions as F, types as T
from pipeline.spark.session import spark, ROOT

LAND = os.path.join(ROOT, "data", "lake", "landing", "openalex")


@F.udf(T.StringType())
def rebuild_abstract(inv):
    if not inv:
        return None
    pos = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(pos))


def main():
    s = spark("bronze_openalex")
    s.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze"); s.sql("CREATE NAMESPACE IF NOT EXISTS lake.silver")
    if os.path.isdir(os.path.join(LAND, "resolve")):
        r = s.read.json(f"{LAND}/resolve/*/*.jsonl.gz", primitivesAsString=False)
        votes = (r.select("inv_id", "n_works", F.explode(F.from_json(F.to_json("votes"), "map<string,struct<n:int,name:string,orcid:string>>")).alias("author_id", "v"))
                 .select("inv_id", "n_works", "author_id", F.col("v.n").alias("votes"), F.col("v.name").alias("author_name"), F.col("v.orcid").alias("orcid")))
        votes.writeTo("lake.bronze.openalex_pi_link").using("iceberg").createOrReplace()
        from pyspark.sql.window import Window
        w = Window.partitionBy("inv_id").orderBy(F.desc("votes"))
        best = (votes.withColumn("rk", F.row_number().over(w)).filter("rk = 1")
                .filter((F.col("votes") >= 2) | (F.col("n_works") == 1)).drop("rk")
                .withColumn("method", F.lit("pmid_authorship_vote")))
        best.writeTo("lake.silver.pi_author").using("iceberg").createOrReplace()
        print("pi links", best.count())
    if os.path.isdir(os.path.join(LAND, "works")):
        w = s.read.json(f"{LAND}/works/*/*.jsonl.gz")
        works = (w.dropDuplicates(["id"]).select(
            F.col("id").alias("work_id"), "doi", "title", "publication_year", "cited_by_count",
            rebuild_abstract(F.from_json(F.to_json("abstract_inverted_index"), "map<string,array<int>>")).alias("abstract"),
            F.col("primary_topic.display_name").alias("topic"), F.col("primary_topic.field.display_name").alias("field"),
            F.expr("transform(authorships, a -> a.author.id)").alias("author_ids")))
        works.writeTo("lake.bronze.openalex_work").using("iceberg").createOrReplace()
        print("works", works.count())
    s.stop()


if __name__ == "__main__":
    main()
