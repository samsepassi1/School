"""Seed the external CultPass database with members, events, bookings, and payments."""

from __future__ import annotations

from data.external import db


MEMBERS = [
    dict(member_id="cp_m_001", full_name="Alice Nguyen",   email="alice@cultpass.test",   tier="plus",    status="active",   city="London"),
    dict(member_id="cp_m_002", full_name="Bob Martinez",   email="bob@cultpass.test",     tier="classic", status="past_due", city="Madrid"),
    dict(member_id="cp_m_003", full_name="Carla Schmidt",  email="carla@cultpass.test",   tier="elite",   status="active",   city="Berlin"),
    dict(member_id="cp_m_004", full_name="Diego Park",     email="diego@cultpass.test",   tier="classic", status="active",   city="Lisbon"),
    dict(member_id="cp_m_005", full_name="Emi Tanaka",     email="emi@cultpass.test",     tier="plus",    status="paused",   city="Tokyo"),
    dict(member_id="cp_m_006", full_name="Frank Li",       email="frank@cultpass.test",   tier="classic", status="active",   city="London"),
]

EVENTS = [
    dict(event_id="cp_e_001", title="Late Modern: Rothko Tour",     venue="Tate Modern",         city="London", starts_at="2026-05-12 19:30", capacity=40,  price_cents=0,    category="art"),
    dict(event_id="cp_e_002", title="Indie Night: The Plagues",     venue="Hoxton Hall",         city="London", starts_at="2026-05-15 20:00", capacity=120, price_cents=0,    category="music"),
    dict(event_id="cp_e_003", title="Hamlet (in the round)",        venue="Donmar Warehouse",    city="London", starts_at="2026-05-20 19:30", capacity=80,  price_cents=1500, category="theatre"),
    dict(event_id="cp_e_004", title="Berlin Sound Lab",             venue="Berghain Kantine",    city="Berlin", starts_at="2026-05-18 21:00", capacity=200, price_cents=0,    category="music"),
    dict(event_id="cp_e_005", title="Late: Almodovar restored",     venue="Cine Doré",           city="Madrid", starts_at="2026-05-22 20:00", capacity=60,  price_cents=0,    category="film"),
    dict(event_id="cp_e_006", title="Chef's Table: Lisbon Tasca",   venue="Tasca da Esquina",    city="Lisbon", starts_at="2026-05-25 19:30", capacity=20,  price_cents=4500, category="food"),
    dict(event_id="cp_e_007", title="In Conversation: Han Kang",    venue="Southbank Centre",    city="London", starts_at="2026-06-02 19:00", capacity=300, price_cents=0,    category="talk"),
]

BOOKINGS = [
    dict(booking_id="cp_b_001", member_id="cp_m_001", event_id="cp_e_001", status="confirmed",  plus_one=1),
    dict(booking_id="cp_b_002", member_id="cp_m_001", event_id="cp_e_007", status="confirmed",  plus_one=0),
    dict(booking_id="cp_b_003", member_id="cp_m_002", event_id="cp_e_005", status="confirmed",  plus_one=0),
    dict(booking_id="cp_b_004", member_id="cp_m_003", event_id="cp_e_004", status="confirmed",  plus_one=1),
    dict(booking_id="cp_b_005", member_id="cp_m_004", event_id="cp_e_006", status="waitlisted", plus_one=0),
    dict(booking_id="cp_b_006", member_id="cp_m_006", event_id="cp_e_002", status="cancelled",  plus_one=0),
    dict(booking_id="cp_b_007", member_id="cp_m_006", event_id="cp_e_003", status="confirmed",  plus_one=1),
]

PAYMENTS = [
    dict(payment_id="cp_p_001", member_id="cp_m_001", amount_cents=1499, kind="subscription", status="captured"),
    dict(payment_id="cp_p_002", member_id="cp_m_002", amount_cents=999,  kind="subscription", status="failed"),
    dict(payment_id="cp_p_003", member_id="cp_m_002", amount_cents=999,  kind="subscription", status="captured"),
    dict(payment_id="cp_p_004", member_id="cp_m_002", amount_cents=999,  kind="subscription", status="captured"),  # duplicate charge!
    dict(payment_id="cp_p_005", member_id="cp_m_003", amount_cents=2999, kind="subscription", status="captured"),
    dict(payment_id="cp_p_006", member_id="cp_m_006", amount_cents=1500, kind="event",        status="captured"),
]


def seed_all(db_path: str | None = None, *, reset: bool = True) -> dict[str, int]:
    if reset:
        db.reset_db(db_path)
    else:
        db.init_db(db_path)
    counts = {
        "CultPassMember":  db.insert_many("CultPassMember",  MEMBERS,  db_path),
        "CultPassEvent":   db.insert_many("CultPassEvent",   EVENTS,   db_path),
        "CultPassBooking": db.insert_many("CultPassBooking", BOOKINGS, db_path),
        "CultPassPayment": db.insert_many("CultPassPayment", PAYMENTS, db_path),
    }
    return counts
