"""Seed sample data for UDA-Hub: accounts, users, prior tickets, and the knowledge base.

The knowledge base ships with 16 articles (4 starter + 12 additional) spanning
account management, billing, technical issues, security, and policy. The rubric
requires at least 14 (4 provided + 10 additional).
"""

from __future__ import annotations

from uda_hub import db


# --- Sample tenants & users -------------------------------------------------

ACCOUNTS = [
    dict(account_id="acc_001", name="Acme Corp",        plan="pro",        status="active",   balance_cents=0),
    dict(account_id="acc_002", name="Globex",           plan="basic",      status="past_due", balance_cents=1999),
    dict(account_id="acc_003", name="Initech",          plan="enterprise", status="active",   balance_cents=0),
    dict(account_id="acc_004", name="Hooli",            plan="free",       status="active",   balance_cents=0),
    dict(account_id="acc_005", name="Soylent",          plan="pro",        status="paused",   balance_cents=0),
]

USERS = [
    dict(user_id="usr_001", account_id="acc_001", email="alice@acme.test",      full_name="Alice Nguyen",     locale="en-US"),
    dict(user_id="usr_002", account_id="acc_002", email="bob@globex.test",      full_name="Bob Martinez",     locale="en-US"),
    dict(user_id="usr_003", account_id="acc_003", email="carla@initech.test",   full_name="Carla Schmidt",    locale="de-DE"),
    dict(user_id="usr_004", account_id="acc_004", email="diego@hooli.test",     full_name="Diego Park",       locale="en-US"),
    dict(user_id="usr_005", account_id="acc_005", email="emi@soylent.test",     full_name="Emi Tanaka",       locale="ja-JP"),
    dict(user_id="usr_006", account_id="acc_001", email="frank@acme.test",      full_name="Frank Li",         locale="en-US"),
]


# --- Prior tickets (so returning customers have history) --------------------

PRIOR_TICKETS = [
    dict(
        ticket_id="tkt_seed_001",
        user_id="usr_001",
        subject="Password reset email not arriving",
        body="I requested a password reset twice and never got the email.",
        status="resolved",
        meta=dict(channel="email", urgency="normal", category="account", sentiment="neutral", routed_to="resolver"),
        messages=[
            ("customer", "I requested a password reset twice and never got the email."),
            ("agent", "Please check your spam folder and ensure alice@acme.test is correct. We have re-sent the reset email."),
            ("customer", "Found it in spam — thanks!"),
        ],
    ),
    dict(
        ticket_id="tkt_seed_002",
        user_id="usr_002",
        subject="Charged twice for May invoice",
        body="My credit card was charged $19.99 twice this month.",
        status="resolved",
        meta=dict(channel="web", urgency="high", category="billing", sentiment="negative", routed_to="resolver"),
        messages=[
            ("customer", "My credit card was charged $19.99 twice this month."),
            ("tool", "process_refund: refunded 1999 cents to acc_002"),
            ("agent", "Refund issued for the duplicate charge. It will appear in 5-10 business days."),
        ],
    ),
    dict(
        ticket_id="tkt_seed_003",
        user_id="usr_003",
        subject="API returns 500 on bulk export",
        body="Our nightly export job has been failing for two days.",
        status="escalated",
        meta=dict(channel="email", urgency="critical", category="technical", sentiment="negative", routed_to="escalation"),
        messages=[
            ("customer", "Our nightly export job has been failing for two days."),
            ("agent", "Escalating to our engineering on-call team. Tracking under INC-4421."),
        ],
    ),
]


# --- Knowledge base ---------------------------------------------------------

KNOWLEDGE_ARTICLES = [
    # ---------------- Starter set (4) -------------------------------------
    dict(
        article_id="kb_001",
        title="How to reset your password",
        category="account",
        tags="password,reset,login,access",
        body=(
            "If you cannot sign in, click 'Forgot password?' on the login screen and "
            "enter the email address on file. You will receive a reset link valid for "
            "30 minutes. If the email does not arrive, check your spam folder, confirm "
            "the address is correct, and ensure noreply@uda-hub.test is allow-listed. "
            "After resetting, you will be signed out of all devices for security."
        ),
    ),
    dict(
        article_id="kb_002",
        title="Refund policy and how to request a refund",
        category="billing",
        tags="refund,billing,charge,policy",
        body=(
            "Refunds are available within 30 days of the original charge for monthly "
            "plans and within 14 days for annual plans, prorated by remaining usage. "
            "Duplicate charges are always refunded in full. Submit a request from "
            "Billing > Invoices, or ask a support agent — automated refunds are "
            "available for amounts under $50."
        ),
    ),
    dict(
        article_id="kb_003",
        title="Update your billing information",
        category="billing",
        tags="billing,credit card,payment method",
        body=(
            "Go to Settings > Billing > Payment Method to add or replace a card. "
            "We accept Visa, Mastercard, AmEx, and Discover. Updates take effect on "
            "your next invoice; pending invoices keep their original method unless "
            "you click 'Retry with new card'."
        ),
    ),
    dict(
        article_id="kb_004",
        title="Upgrade or downgrade your account plan",
        category="account",
        tags="plan,upgrade,downgrade,subscription",
        body=(
            "Plan changes take effect immediately. Upgrades are charged prorated for "
            "the remainder of the billing cycle. Downgrades schedule the new plan to "
            "begin at the next renewal — your current plan continues until then. "
            "Annual plans must finish their term before downgrading to monthly."
        ),
    ),
    # ---------------- Additional set (12) ---------------------------------
    dict(
        article_id="kb_005",
        title="Enable two-factor authentication (2FA)",
        category="security",
        tags="2fa,security,login,auth",
        body=(
            "From Settings > Security, choose 'Add authenticator app' and scan the QR "
            "code with Google Authenticator, 1Password, or any TOTP app. Save the "
            "10 backup codes shown — you will need them if you lose your device. "
            "SMS-based 2FA is supported but discouraged for accounts with admin access."
        ),
    ),
    dict(
        article_id="kb_006",
        title="Cancel your subscription",
        category="billing",
        tags="cancel,subscription,churn",
        body=(
            "You can cancel any time from Settings > Billing > Cancel Plan. Your "
            "subscription remains active until the end of the current billing cycle. "
            "After cancellation we retain your data for 90 days; reactivate within "
            "that window to restore everything. Annual plans are non-refundable except "
            "during the 14-day refund window."
        ),
    ),
    dict(
        article_id="kb_007",
        title="Pause your subscription temporarily",
        category="billing",
        tags="pause,subscription,vacation,hold",
        body=(
            "Pro and Enterprise customers can pause billing for up to 90 days per "
            "calendar year. While paused, your account is read-only and we do not "
            "charge you. Resume at any time from Settings > Billing — paused months "
            "do not count toward annual plan terms."
        ),
    ),
    dict(
        article_id="kb_008",
        title="Why my payment failed",
        category="billing",
        tags="payment,failed,decline,card",
        body=(
            "The most common reasons for declined payments are: insufficient funds, "
            "expired card, mismatched billing address, or a bank flagging the charge "
            "as suspicious. We retry failed charges three times over five days. "
            "Update your card under Settings > Billing or contact your bank to "
            "approve the charge."
        ),
    ),
    dict(
        article_id="kb_009",
        title="Change the email address on your account",
        category="account",
        tags="email,change,profile",
        body=(
            "Go to Settings > Profile and click 'Change email'. We will send a "
            "verification link to the new address — until you click it, sign-in still "
            "uses the old email. Workspace owners must transfer ownership before "
            "deleting the original email."
        ),
    ),
    dict(
        article_id="kb_010",
        title="Export or delete your data (GDPR / CCPA)",
        category="privacy",
        tags="gdpr,ccpa,export,delete,privacy",
        body=(
            "From Settings > Privacy you can request a full export of your data "
            "(JSON + CSV) or schedule account deletion. Exports are emailed within "
            "72 hours. Deletion runs after a 7-day cooling-off period and is "
            "irreversible — backups are purged within 30 days."
        ),
    ),
    dict(
        article_id="kb_011",
        title="App is slow or freezing on mobile",
        category="technical",
        tags="performance,mobile,slow,freeze",
        body=(
            "First, force-quit and reopen the app. If the issue persists: clear cache "
            "(Settings > Storage > Clear cache), update to the latest app version, "
            "and ensure your device has at least 500MB free. Older devices may need "
            "to disable 'Animated avatars' under Display settings."
        ),
    ),
    dict(
        article_id="kb_012",
        title="Connect a third-party integration (Slack, Zapier, Salesforce)",
        category="technical",
        tags="integration,slack,zapier,salesforce,api",
        body=(
            "Open Settings > Integrations and select the provider. You will be sent "
            "to the provider's OAuth consent screen — approve the requested scopes "
            "and you will be redirected back. For Salesforce, you must be a System "
            "Administrator; for Slack, you must have permission to install apps."
        ),
    ),
    dict(
        article_id="kb_013",
        title="Invite teammates and manage seats",
        category="account",
        tags="invite,team,seats,members",
        body=(
            "Go to Settings > Team > Invite member, enter their email, and pick a "
            "role (Owner, Admin, Member, Viewer). Invitations expire after 7 days. "
            "Adding a seat above your plan limit triggers prorated billing on your "
            "next invoice."
        ),
    ),
    dict(
        article_id="kb_014",
        title="Recover a deleted account",
        category="account",
        tags="recover,delete,restore",
        body=(
            "Deleted accounts can be restored within 90 days from the deletion email "
            "we sent you (subject: 'Your UDA-Hub account has been deleted'). Click "
            "'Restore account' in that email or contact support with your former "
            "account email. After 90 days, recovery is no longer possible."
        ),
    ),
    dict(
        article_id="kb_015",
        title="API rate limits and 429 errors",
        category="technical",
        tags="api,rate limit,429,errors",
        body=(
            "Default rate limits are 60 requests/minute on Free, 600/minute on Pro, "
            "and 6000/minute on Enterprise. A 429 response includes a 'Retry-After' "
            "header — respect it with exponential backoff. Sustained 429s on "
            "Enterprise plans usually mean a single API key is hot — split traffic "
            "across keys."
        ),
    ),
    dict(
        article_id="kb_016",
        title="Suspicious login or compromised account",
        category="security",
        tags="security,compromise,suspicious,hack",
        body=(
            "If you see a sign-in you don't recognise: (1) change your password "
            "immediately, (2) sign out all sessions from Settings > Security > "
            "Active sessions, (3) enable 2FA, and (4) review recent activity in "
            "Settings > Security > Audit log. Contact support if you see "
            "unauthorised data exports or admin role changes."
        ),
    ),
]


def seed_all(db_path: str | None = None, *, reset: bool = True) -> dict[str, int]:
    """Populate every table. Returns row counts."""
    if reset:
        db.reset_db(db_path)
    else:
        db.init_db(db_path)

    counts: dict[str, int] = {}
    counts["Account"] = db.insert_many("Account", ACCOUNTS, db_path)
    counts["User"] = db.insert_many("User", USERS, db_path)
    counts["Knowledge"] = db.insert_many("Knowledge", KNOWLEDGE_ARTICLES, db_path)

    # tickets + metadata + messages
    ticket_rows, meta_rows, message_rows = [], [], []
    for t in PRIOR_TICKETS:
        ticket_rows.append(
            dict(
                ticket_id=t["ticket_id"],
                user_id=t["user_id"],
                subject=t["subject"],
                body=t["body"],
                status=t["status"],
            )
        )
        meta_rows.append(
            dict(
                ticket_id=t["ticket_id"],
                channel=t["meta"]["channel"],
                urgency=t["meta"]["urgency"],
                category=t["meta"]["category"],
                sentiment=t["meta"]["sentiment"],
                confidence=0.9,
                routed_to=t["meta"]["routed_to"],
                extra_json=None,
            )
        )
        for role, content in t["messages"]:
            message_rows.append(
                dict(
                    ticket_id=t["ticket_id"],
                    role=role,
                    author=t["user_id"] if role == "customer" else role,
                    content=content,
                )
            )
    counts["Ticket"] = db.insert_many("Ticket", ticket_rows, db_path)
    counts["TicketMetadata"] = db.insert_many("TicketMetadata", meta_rows, db_path)
    counts["TicketMessage"] = db.insert_many("TicketMessage", message_rows, db_path)
    return counts
