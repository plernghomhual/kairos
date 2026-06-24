# Kairos — Design Partner Outreach Emails + Target List

Companion to `design_partner_ask.md` (attach it or paste its content as the follow-up).
Rule for all variants: lead with the question, not the product. Short enough to read on a phone. One ask, one click.

---

## Email A — Cold, to a Head of Execution / Trading (equities/futures desk)

**Subject:** Can you split your slippage into your impact vs. the venue's congestion?

Hi [Name],

Quick question that most TCA can't answer: when your fills come in worse than expected, how much of that is your own market impact — and how much is your order sitting in a gateway or venue queue while the market moved away from you?

Tick-data TCA can't separate the two because it never sees your own send→ack→fill timing. We measure that directly, from your own FIX logs, inside your perimeter — nothing leaves your building.

We're selecting 2–3 design partners for a free, no-commitment analysis: you give us one anonymized telemetry export (six fields, no order contents, no strategy, venue names can be hashed), we return a per-venue decomposition of your slippage into own-impact vs. infrastructure congestion, with statistical confidence.

If congestion explains a real share, you've found cost you can route around. If it doesn't, you've spent an hour confirming your infra is clean. Either answer is worth the hour.

Worth a 20-minute call?

[Signature]

---

## Email B — Warm intro version (forwardable)

**Subject:** Intro — measuring execution infrastructure cost from your own FIX logs

Hi [Name],

[Intro-giver] suggested I reach out. One line on what we do: we measure how much of a desk's slippage comes from infrastructure congestion (gateway dwell, queue residency, rejects) versus the desk's own market impact — measured from your own FIX timing, locally, with no data leaving your perimeter.

We're looking for a design partner to run this on a few volatile sessions. Free, anonymized, one-page result back to you. The data ask is six plumbing fields per venue per minute — no order contents, no counterparties, no strategy.

20 minutes to see if it's interesting?

[Signature]

---

## Email C — Crypto-native trading firm variant (fastest expected replies)

**Subject:** Splitting your slippage: your impact vs. exchange/gateway congestion

Hi [Name],

Crypto venues make this problem worse than equities: matching-engine congestion, API rate limits, and settlement delays eat PnL in ways tick-data analytics never show — because they can't see your own order→ack→fill timing.

We measure it directly from your own gateway logs, locally — raw data never leaves your infra. We're picking a couple of design partners for a free analysis: you export six anonymized timing/throughput fields per venue per minute, we hand back a decomposition of your slippage into own-impact vs. venue/infra congestion, per venue, with confidence intervals.

You already feel this on volatile days. We can tell you what it costs you.

20 minutes this week?

[Signature]

---

## Target list — picked for reply likelihood, not prestige

Honest note: I cannot verify individual inboxes — find the named role on LinkedIn and use the firm's standard email format. Do NOT blast all at once; send 3–5, iterate on what bounces back.

### Tier 1 — most likely to reply fast (transparency-mission or crypto-native)

| Firm | Who | Why they'd reply |
|---|---|---|
| **Proof Trading** (NYC, agency broker) | Daniel Aisen (CEO, ex-IEX) or Allison Bishop | Built the firm ON execution transparency; they blog their own research openly; your pitch is literally their worldview. Highest reply probability on this list. |
| **Themis Trading** (NJ, agency broker) | Joe Saluzzi / Sal Arnuk (co-founders) | Famous market-structure critics; answer outreach about plumbing/fairness constantly; media-active. Even a "no" comes with useful opinion. |
| **Wintermute** (London, crypto MM) | Head of Trading / CTO office | Crypto-native, fast-moving, feels venue congestion daily, no legacy procurement wall. Kairos's existing crypto plumbing is credible here. |
| **GSR** (crypto MM) | Head of Execution | Same logic as Wintermute; institutional-facing so best-ex framing lands. |
| **B2C2** (London, crypto OTC MM) | eTrading / quant exec lead | OTC + venue connectivity = lives the dwell/reject problem; mid-size, approachable. |

### Tier 2 — mid-tier prop/quant desks (the actual target persona)

| Firm | Who | Why |
|---|---|---|
| **Geneva Trading** (Chicago prop) | Head of Trading / CTO | Mid-tier futures prop; sophisticated enough to have FIX infra, small enough not to have built this. |
| **Belvedere Trading** (Chicago options MM) | Trading Technology lead | Mid-tier, tech-forward culture, known to engage outside vendors. |
| **XR Trading** (Chicago) | Head of Technology | Mid-tier futures MM; latency-aware but not Citadel-scale instrumentation. |
| **Old Mission Capital** (Chicago) | Head of Execution / Quant | Multi-asset mid-tier MM; exactly the band where the edge is largest. |
| **Quantlab** (Houston) | CTO office | Quant trading, mid-tier, engineering-led. |

### Tier 3 — backup / institutional angle

| Firm | Who | Why |
|---|---|---|
| **CoinRoutes** (crypto execution/SOR vendor) | CEO Dave Weisberger | Not a desk — a potential channel partner: their SOR could consume your congestion signal. Replies to market-structure conversations publicly. |
| **Talos** (institutional crypto infra) | Product / partnerships | Channel partner angle, same logic. |
| **A regional agency broker-dealer** (e.g., one your network can intro) | Head of Best-Ex | Best-ex is their regulatory budget line; slower to reply but the mandate is real. |

### Sequencing
1. **Send first (this week):** Proof Trading, Themis, Wintermute — highest reply odds, fastest feedback on the pitch itself.
2. **Second wave:** 2–3 of the Chicago mid-tier shops with Email A.
3. **Use replies to refine** the ask before touching Tier 3 / channel partners.

---

## LinkedIn DM variants (no-call path — fully async)

All CTAs are reply-only. No calls anywhere in this funnel: they reply → you send the one-pager + data spec → they export → you return the report. Every step is a message.

### DM-1 — Connection request note (300-char limit, use with request)

> Quick question most TCA can't answer: how much of your slippage is your own impact vs. your order stuck in a gateway/venue queue? We measure it from your own FIX timing, locally — data never leaves your infra. Happy to share the 1-pager if interesting.

### DM-2 — Full message (after connect, or InMail) — equities/prop version

> Hi [Name] — one question: when fills come in worse than expected, can you currently split that into your own market impact vs. infrastructure congestion (gateway dwell, queue residency, rejects)? Tick-data TCA can't — it never sees your send→ack→fill timing.
>
> We measure it directly from your own logs, inside your perimeter, nothing leaves your building. Looking for 2–3 design partners: you export six anonymized plumbing fields (no order contents, no strategy, venues can be hashed), we return a per-venue decomposition of slippage into own-impact vs. congestion, with confidence intervals. Free, no commitment, no call needed — everything async.
>
> Want the one-page data spec?

### DM-3 — Crypto-native version

> Hi [Name] — crypto venues make this worse than equities: matching-engine congestion, rate limits, settlement delays eat PnL in ways tick analytics never show, because they can't see your own order→ack→fill timing. We measure it from your gateway logs, locally — raw data never leaves your infra.
>
> Picking a couple of design partners for a free async analysis: six anonymized timing fields per venue per minute in, per-venue slippage decomposition (your impact vs. venue/infra congestion) out. No call required — all over messages.
>
> Send you the data spec?

### DM-4 — Themis/Proof transparency-flavored version

> Hi [Name] — long-time reader of your market-structure work. We're building independent measurement of execution infrastructure cost: splitting slippage into own-impact vs. gateway/venue congestion, measured from the desk's own FIX timing rather than inferred from tick data. Local-first — nothing leaves the desk's perimeter.
>
> Looking for a design partner to run it on a few volatile sessions, free, fully async. Given your writing on venue transparency, wanted to ask you first. Can I send the one-pager?

### Async-funnel rules
1. Every CTA = "want the one-pager / data spec?" — never "can we call."
2. If THEY ask for a call, that's success — decide then; you can offer a Loom/recorded walkthrough or a written Q&A instead.
3. Reply → send `design_partner_ask.md` content + the six-field spec.
4. Deliverable back to them = one-page written report. The entire relationship can stay in text.

### Optional inbound play (good fit for async temperament)
Publish the method publicly — a short post showing the decomposition on synthetic data (the validated harness output) with the honest framing "looking for one desk to test on real telemetry." Proof Trading built credibility exactly this way (public research). Inbound replies cost zero social energy.

---

### Why crypto-first is honest, not a cop-out
The strategy targets equities/futures desks long-term, but crypto trading firms (a) reply in days not quarters, (b) have no procurement bureaucracy, (c) suffer venue congestion *worse*, and (d) match Kairos's existing codebase credibility. A crypto desk's gateway logs satisfy the §9 thesis gate just as well as an equities FIX log — committed telemetry is committed telemetry.
