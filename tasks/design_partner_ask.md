# Kairos — Design Partner Request (1 page)

**To:** Head of Execution / Head of Trading / Quant Execution Lead
**From:** Kairos (Ultraqard Technologies)
**Ask:** 60 minutes + one anonymized data sample. No software installed. No data leaves your perimeter.

---

## The question we want to answer for you

When your fills come in worse than expected, how much of that slippage is **your own market impact** (you moved the market) versus **infrastructure congestion** (your order sat in a gateway/venue queue while the market moved away from you)?

Today's TCA can't separate these. It reconstructs the market from public tick data — it never sees your own gateway's send→ack→fill timing. We measure that directly. The congestion slice is often the part you can actually fix (route differently, throttle, switch session) — but only if you can see it.

## What we need from you (the smallest possible sample)

One **anonymized** export covering a few volatile trading sessions. Per venue, per minute, six numbers:

| # | Field | Source (standard FIX) |
|---|-------|------------------------|
| 1 | venue / gateway id (can be a hashed label) | session `SenderCompID`/`TargetCompID` |
| 2 | minute bucket | `SendingTime` |
| 3 | order throughput (msgs/sec) | message count per session |
| 4 | gateway dwell (send→ack, ack→fill, seconds) | `SendingTime` vs ACK/fill `TransactTime` |
| 5 | in-flight queue depth | open orders between send and terminal state |
| 6 | rejection rate | `ExecReport(150=8)`, `OrdRejReason`, session rejects |

Plus, for the same minutes, your own realized slippage/markout (which you already compute).

**No order contents. No counterparties. No strategy. No prices we don't already have.** Venue names can be hashed before they reach us. The six fields are pure plumbing telemetry — they reveal nothing about what you traded or why.

## What you get back (free, no commitment)

A one-page decomposition of your slippage into **own-impact vs. gateway-congestion**, per venue, with statistical confidence — using the validated harness in `congestion_test_harness.py`. If congestion explains a material, independent share of your slippage, you've found cost you can route around. If it doesn't, you've spent an hour and confirmed your infra is clean. Either result is worth the hour.

## Why you, why now

We're selecting a small number of design partners. We're not asking you to buy anything. We're asking to prove — on your data, in your building — whether this measurement is worth productizing. You keep the analysis regardless.

---

### Who to send this to (target list, in priority order)

1. **Mid-tier systematic hedge funds / prop trading firms** — sophisticated enough to have FIX infra and care about best-ex, not so large they've already built bespoke gateway instrumentation. **Best fit.**
2. **Mid-tier market makers** — congestion directly costs them; they feel it daily.
3. **Agency/execution broker-dealers** — best-ex is a regulatory mandate (MiFID II RTS 27/28, SEC Rule 605/606), so the budget already exists.
4. **Avoid first:** Tier-1 banks and the top quant shops (Citadel, Jane Street, HRT) — they already instrument their own latency; your edge is smallest there.

**Decision-maker / signer:** Head of Execution or Head of Trading. Technical validator: their lead execution quant. CTO only if the data-access question escalates.

**Warm-path framing:** lead with the *question* ("can you split impact from congestion today?"), not the product. The question sells; the product follows the answer.
