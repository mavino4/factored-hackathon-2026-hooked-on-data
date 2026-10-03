-- Simulated "core banking" database (separate from the app's own database).
-- Loaded from the Datathon customers/products tables by scripts/load_bank_db.py.
--
-- Access model: the app connects as the read-only role bank_reader and, in every
-- transaction, sets app.subject to the authenticated user's token `sub`.
-- Row-Level Security then exposes ONLY the rows of the customer linked to that
-- subject. Without app.subject (or with an unlinked subject) every query returns
-- zero rows, even a SELECT without WHERE.
--
-- Personal data is minimized on purpose: no documents, last names, birth dates,
-- emails, phones, addresses, income or credit scores are stored here, and card and
-- account numbers keep only their last 4 digits.

CREATE SCHEMA IF NOT EXISTS bank;

CREATE TABLE IF NOT EXISTS bank.customers (
    customer_id      text PRIMARY KEY,
    first_name       text NOT NULL,
    country          text,
    city             text,
    segment          text,
    customer_status  text,
    tenure_years     numeric(6, 2)
);

CREATE TABLE IF NOT EXISTS bank.products (
    product_id             text PRIMARY KEY,
    customer_id            text NOT NULL REFERENCES bank.customers (customer_id),
    product_type           text NOT NULL,
    product_number_last4   char(4),
    currency               char(3) NOT NULL,
    -- Meaning depends on product_type: accounts = available funds,
    -- credit cards = amount owed, loans = outstanding debt.
    current_balance        numeric(18, 2) NOT NULL,
    -- Credit cards = credit limit, loans = amount granted.
    credit_limit           numeric(18, 2),
    interest_rate          numeric(6, 2),
    opening_date           date,
    expiration_date        date,
    product_status         text NOT NULL,
    days_past_due          integer,
    has_linked_app         boolean,
    last_transaction_date  date,
    available_credit       numeric(18, 2) GENERATED ALWAYS AS (
        CASE WHEN product_type = 'Tarjeta Crédito' AND credit_limit IS NOT NULL
             THEN credit_limit - current_balance END
    ) STORED
);
CREATE INDEX IF NOT EXISTS ix_products_customer ON bank.products (customer_id);

-- Links an identity-provider user (token `sub`) to a bank customer.
CREATE TABLE IF NOT EXISTS bank.customer_logins (
    subject      text PRIMARY KEY CHECK (subject <> ''),
    customer_id  text NOT NULL REFERENCES bank.customers (customer_id)
);
CREATE INDEX IF NOT EXISTS ix_customer_logins_customer ON bank.customer_logins (customer_id);

-- Row-Level Security: FORCE also applies it to the table owner.
-- (Superusers always bypass RLS; the app never connects as one.)
ALTER TABLE bank.customer_logins ENABLE ROW LEVEL SECURITY;
ALTER TABLE bank.customer_logins FORCE ROW LEVEL SECURITY;
ALTER TABLE bank.customers ENABLE ROW LEVEL SECURITY;
ALTER TABLE bank.customers FORCE ROW LEVEL SECURITY;
ALTER TABLE bank.products ENABLE ROW LEVEL SECURITY;
ALTER TABLE bank.products FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS own_login ON bank.customer_logins;
CREATE POLICY own_login ON bank.customer_logins FOR SELECT
    USING (subject = current_setting('app.subject', true));

-- A scalar subquery (`=`, not `IN`): a subject links to at most one customer (it is the
-- primary key), and Postgres evaluates it once and uses ix_products_customer. With `IN`
-- it scanned all 400k products on every query (seconds when the cache is cold). No
-- linked customer: the subquery is NULL and no row matches.
DROP POLICY IF EXISTS own_customer ON bank.customers;
CREATE POLICY own_customer ON bank.customers FOR SELECT
    USING (customer_id = (SELECT l.customer_id FROM bank.customer_logins l
                          WHERE l.subject = current_setting('app.subject', true)));

DROP POLICY IF EXISTS own_products ON bank.products;
CREATE POLICY own_products ON bank.products FOR SELECT
    USING (customer_id = (SELECT l.customer_id FROM bank.customer_logins l
                          WHERE l.subject = current_setting('app.subject', true)));

-- Read-only application role (created by the loader with its password).
GRANT USAGE ON SCHEMA bank TO bank_reader;
GRANT SELECT ON bank.customers, bank.products, bank.customer_logins TO bank_reader;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON bank.customers, bank.products, bank.customer_logins
    FROM bank_reader;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
