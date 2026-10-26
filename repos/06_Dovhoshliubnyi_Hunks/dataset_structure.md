# Database Schema: Pull Request and Repository Analytics

## Tables

### `pr_commits`
* **sha** (varchar) - *Primary Key*
* **pr_id** (integer) - *Foreign Key to `pull_request.id`*
* **author** (varchar)
* **committer** (varchar)
* **message** (text)

### `user`
* **id** (integer) - *Primary Key*
* **login** (varchar)
* **followers** (integer)
* **following** (integer)
* **created_at** (timestamp)

### `pr_commit_details`
* **sha** (varchar) - *Primary Key Part 1, Foreign Key to `pr_commits.sha`*
* **filename** (varchar) - *Primary Key Part 2*
* **pr_id** (integer) - *Foreign Key to `pull_request.id`*
* **author** (varchar)
* **committer** (varchar)
* **message** (text)
* **commit_stats_total** (integer)
* **commit_stats_additions** (integer)
* **commit_stats_deletions** (integer)
* **status** (varchar)
* **additions** (integer)
* **deletions** (integer)
* **changes** (integer)
* **patch** (text)

### `pr_timeline`
* **pr_id** (integer) - *Foreign Key to `pull_request.id`*
* **event** (varchar)
* **commit_id** (varchar)
* **created_at** (timestamp)
* **actor** (varchar)
* **assignee** (varchar)
* **label** (varchar)
* **message** (text)

### `pull_request`
* **id** (integer) - *Primary Key*
* **number** (integer)
* **title** (varchar)
* **body** (text)
* **agent** (varchar)
* **user_id** (integer) - *Foreign Key to `user.id`*
* **user** (varchar)
* **state** (varchar)
* **created_at** (timestamp)
* **closed_at** (timestamp)
* **merged_at** (timestamp)
* **repo_id** (integer) - *Foreign Key to `repository.id`*
* **repo_url** (varchar)
* **html_url** (varchar)

### `related_issue`
* **pr_id** (integer) - *Primary Key Part 1, Foreign Key to `pull_request.id`*
* **issue_id** (integer) - *Primary Key Part 2, Foreign Key to `issue.id`*
* **source** (varchar)

### `repository`
* **id** (integer) - *Primary Key*
* **url** (varchar)
* **license** (varchar)
* **full_name** (varchar)
* **language** (varchar)
* **forks** (integer)
* **stars** (integer)

### `pr_task_type`
* **pr_id** (integer) - *Foreign Key to `pull_request.id`*
* **type** (varchar)

### `pr_reviews`
* **id** (integer) - *Primary Key*
* **pr_id** (integer) - *Foreign Key to `pull_request.id`*
* **user** (varchar)
* **state** (varchar)
* **submitted_at** (timestamp)
* **body** (text)

### `pr_review_comments`
* **id** (integer) - *Primary Key*
* **pull_request_review_id** (integer) - *Foreign Key to `pr_reviews.id`*
* **user** (varchar)
* **diff_hunk** (text)
* **path** (varchar)
* **position** (integer)
* **original_position** (integer)
* **commit_id** (varchar)
* **original_commit_id** (varchar)
* **body** (text)
* **pull_request_url** (varchar)
* **created_at** (timestamp)
* **updated_at** (timestamp)
* **in_reply_to_id** (integer) - *Self-referencing Foreign Key to `pr_review_comments.id`*

### `pr_comments`
* **id** (integer) - *Primary Key*
* **pr_id** (integer) - *Foreign Key to `pull_request.id`*
* **user** (varchar)
* **user_id** (integer) - *Foreign Key to `user.id`*
* **created_at** (timestamp)
* **body** (text)

### `issue`
* **id** (integer) - *Primary Key*
* **number** (integer)
* **title** (varchar)
* **body** (text)
* **user** (varchar)
* **state** (varchar)
* **created_at** (timestamp)
* **closed_at** (timestamp)
* **html_url** (varchar)

---

## Key Relationships (Entity-Relationship)

* `pull_request.user_id` $\rightarrow$ `user.id` (Many-to-One / 0..1)
* `pull_request.repo_id` $\rightarrow$ `repository.id` (Many-to-One / 0..1)
* `pr_commits.pr_id` $\rightarrow$ `pull_request.id` (Many-to-One / 0..1)
* `pr_commit_details.sha` $\rightarrow$ `pr_commits.sha` (Many-to-One)
* `pr_commit_details.pr_id` $\rightarrow$ `pull_request.id` (Many-to-One / 0..1)
* `related_issue.pr_id` $\rightarrow$ `pull_request.id` (Many-to-One / 0..1)
* `related_issue.issue_id` $\rightarrow$ `issue.id` (Many-to-One / 0..1)
* `pr_timeline.pr_id` $\rightarrow$ `pull_request.id` (Many-to-One / 0..1)
* `pr_task_type.pr_id` $\rightarrow$ `pull_request.id` (Many-to-One / 0..1)
* `pr_reviews.pr_id` $\rightarrow$ `pull_request.id` (Many-to-One / 0..1)
* `pr_review_comments.pull_request_review_id` $\rightarrow$ `pr_reviews.id` (Many-to-One / 0..1)
* `pr_review_comments.in_reply_to_id` $\rightarrow$ `pr_review_comments.id` (Self-Referencing Many-to-One / 0..1)
* `pr_comments.pr_id` $\rightarrow$ `pull_request.id` (Many-to-One / 0..1)
* `pr_comments.user_id` $\rightarrow$ `user.id` (Many-to-One / 0..1)
