-- Users / Accounts
WITH users AS (
    SELECT * FROM (
        VALUES
            (1, 'acct_1'),
            (2, 'acct_1'),
            (3, 'acct_2'),
            (4, 'acct_2'),
            (5, 'acct_3'),
            (6, 'acct_4'),
            (7, 'acct_5'),
            (8, 'acct_6'),
            (9, 'acct_7'),
            (10, 'acct_8')
    ) AS t(user_id, account_id)
),

randomized AS (
    SELECT
        user_id,
        account_id,

        -- 👇 now using account_id ensures same variant per account
        abs(hash(account_id || 'exp_attach_toggle_v1')) % 100 AS bucket

    FROM users
)

SELECT
    user_id,
    account_id,
    bucket,
    CASE
        WHEN bucket < 50 THEN 'control'
        ELSE 'treatment'
    END AS variant
FROM randomized
ORDER BY account_id, user_id;