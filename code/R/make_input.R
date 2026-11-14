d <- read.csv("../data/expert_ratings_long.csv", stringsAsFactors = FALSE)
out <- data.frame(expert = d$rater, question = d$question_id, model = d$system,
                  domain = d$domain, score = d$score, role = d$rater_profession, qid = d$question_id)
write.csv(out, "long_with_global_qid.csv", row.names = FALSE)
cat("wrote long_with_global_qid.csv:", nrow(out), "rows\n")
