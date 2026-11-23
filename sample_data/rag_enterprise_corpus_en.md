# Enterprise RAG Evaluation Corpus (Fictional)

All company names, policies, IDs, and figures in this corpus are fictional. Ingest this Markdown file as a whole, or use the JSONL version to ingest each document separately.

---

## DOC-001 — Support Request Handling Policy

### 1. Scope and support hours

Sao Mai Digital provides an order management and customer care platform for businesses. Standard Support operates Monday through Friday, 08:00–18:00 Vietnam time (UTC+7), excluding company-observed holidays. Customers on the Priority plan may report critical incidents through the 24/7 on-call line. Initial response times are service targets, not commitments to resolve an issue within that time.

### 2. Priority levels

| Level | Description | Initial response target — Priority plan |
|---|---|---:|
| P1 — Critical | The core service is unavailable to most users and no practical workaround exists. | 30 minutes |
| P2 — High | An important function is substantially affected, but some service remains or a limited workaround exists. | 2 business hours |
| P3 — Normal | A non-critical feature is impaired or the customer needs configuration or usage guidance. | 1 business day |
| P4 — Low | A general question, minor UI issue, or feature request. | 3 business days |

Priority is assigned based on business impact and available workarounds, not only on the number of users reporting an issue. Customers may request a priority review and should provide the affected workflow, number of users, start time, and workarounds already attempted.

### 3. Information required in a support ticket

When submitting a request, an administrator should provide:
- Organization name and customer ID.
- Product, environment (production or sandbox), and affected area.
- Start time, time zone, and reproduction steps.
- Error code or correlation ID, if available.
- Number of affected users and workarounds attempted.

Do not send passwords, MFA codes, payment card numbers, or sensitive end-customer personal data through ordinary email or support tickets. If Support needs sensitive data for an investigation, it will direct the customer to an approved secure upload channel.

---

## DOC-002 — Data Storage, Export, and Deletion Policy

### 1. Hosting region

Sao Mai Digital production data is primarily stored in Singapore. Some auxiliary services, such as email delivery, error monitoring, and support tooling, may process limited metadata in other regions. A customer requiring data to remain in a single country must have a data residency addendum signed by both parties; selecting a region in the admin interface does not by itself create a legal residency commitment.

### 2. Backups and recovery

> COPIED FROM PDF — Page 4/18 — SAO MAI DIGITAL — INTERNAL USE

Production databases are backed up daily with encryption. Operational backups are retained for up to 35 days and then expire automatically. Backups support service recovery and are not a long-term customer archive. Any recovery time objective (RTO) or recovery point objective (RPO) is specified in the applicable service agreement; this document sets no universal value.

To export data, customers submit a ticket through the Support Portal and specify the date range, data types, and authorized recipient. Format and delivery time depend on data volume and the service addendum. A request is complete only after the customer receives delivery confirmation.

### 3. End of contract

After a contract ends, customers have 45 days to request a data export unless the contract specifies another period. After that period, production data is deleted within 30 days. Copies remaining in backups expire under the backup lifecycle, no later than 35 days after each copy was created. Legal retention duties or lawful orders may create exceptions.

> COPIED FROM PDF — Page 5/18 — SAO MAI DIGITAL — INTERNAL USE

---

## DOC-003 — Service Plans and Usage Fees

### 1. Platform plans

| Plan | Monthly subscription | Included users | Support | Audit log retention |
|---|---:|---:|---|---:|
| Basic | VND 12,000,000 | 25 | Standard | 30 days |
| Growth | VND 28,000,000 | 100 | Standard | 90 days |
| Enterprise | Quoted | As specified in contract | Priority if stated in the order form | 365 days |

Prices exclude VAT and implementation fees. The Enterprise plan includes Priority Support only if the signed order form expressly states that entitlement. Audit logs may be retained longer under a separate addendum.

### 2. Overage and payment

Each user above the plan limit costs VND 180,000 per user per month, based on the highest number of active users during the month. Overage fees appear on the following month’s invoice. Subscription invoices are issued at the start of each month and are due in 30 days unless the order form says otherwise. If a customer disputes part of an invoice, the undisputed portion remains payable.

### 3. Renewal

Contracts automatically renew for the term stated in the order form. Customers must give notice of non-renewal at least 60 days before the renewal date unless the signed contract specifies a different period. A plan change takes effect only when confirmed in an amended order form or another writing accepted by both parties.

---

## DOC-004 — API Integration Runbook

### 1. Authentication and limits

The API uses a separate bearer token for each environment. Do not use a sandbox token in production. The default limit is 120 requests per minute per organization; some endpoints have lower limits specified in their endpoint documentation. When the limit is exceeded, the API returns HTTP 429 and a `Retry-After` header indicating how many seconds to wait.

### 2. Webhook configuration example

```json
{
  "event": "order.updated",
  "endpoint": "https://merchant.example/webhooks/orders",
  "retries": 5,
  "retry_delay_seconds": 30,
  "signature_header": "X-SM-Signature"
}
```

A webhook is retried up to 5 times, 30 seconds apart. The HMAC signature is computed over the raw request body; parsing and re-serializing the JSON before verification may cause the signature check to fail. The receiving endpoint must return a 2xx status within 10 seconds. After the final failed attempt, the event moves to a dead-letter queue and an administrator must request a resend in the Console.

### 3. Retry example

```python
import time
import requests

def get_order(url, token):
    for attempt in range(5):
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        wait_seconds = int(response.headers.get("Retry-After", "2"))
        time.sleep(wait_seconds)
    raise RuntimeError("API rate limit persisted after five attempts")
```

Never write tokens to logs. Before sharing logs with Support, redact tokens, customer email addresses, and order contents. A 429 indicates a rate limit; a 401 usually means the token is missing, invalid, or revoked.

---

## DOC-005 — New Employee Onboarding Procedure

### A. Before the first working day

The manager opens a request in the HR portal at least 3 business days before the start date. IT creates an account after receiving the employee ID, department, manager, work location, and approved access group. New accounts do not have production access by default.

### B. On the first day

1. The employee signs in with the company account and enrolls in MFA.
2. The manager confirms the device was received and the security awareness course is complete.
3. The system owner separately approves each sensitive permission.
4. IT records completion in the onboarding ticket.

If approval is missing, the employee receives basic access only. Do not share accounts or use a colleague’s account to speed up onboarding.

### C. Termination or department transfer

When possible, the manager submits a change request before its effective date. For an unexpected termination, the manager contacts IT as soon as the decision takes effect. IT disables company sign-in within 4 business hours after receiving an authenticated request; the application owner removes SaaS permissions during the same business day. Business data is not automatically deleted just because an account is disabled.

---

## DOC-006 — Web Extract: Device and Security Policy

### Managed devices

Devices used to access internal data must be registered in the device management system, have disk encryption enabled, lock the screen after no more than 5 minutes of inactivity, and install critical security patches within 14 days of release. Personal devices may access company email only through a managed application and must not download customer data.

### Reporting a security event

Anyone who finds a phishing email, loses a device, or suspects information has been exposed must report it immediately using the “Report Security Event” button on the Intranet. If a device is lost after hours, call the internal on-call number +84 28 5550 0199. Do not wipe the device before Security Operations gives instructions, unless device management requires a remote wipe to prevent unauthorized access.

### Extracted footer

Sao Mai Digital | Employee portal: https://intranet.saomai.example/security | Policy version: 2025-02-01 | Page 1 / 7

The URL and company details are fictional. Footer text may be repeated during PDF/HTML extraction.
