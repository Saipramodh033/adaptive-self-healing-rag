# Payment Failures — Troubleshooting Guide

**Source:** troubleshooting/payment_failures.md
**Last Updated:** January 2025

---

## Overview

Payment failures at checkout are frustrating but usually quick to resolve. This guide covers the most common causes and step-by-step fixes.

---

## Card Declined

### "Payment Declined" or "Transaction Failed" message

**Common causes:**

1. **Insufficient balance:** Your card does not have enough funds for the order total
2. **Daily transaction limit reached:** Most banks set daily UPI/card limits (typically Rs. 1–5 lakh)
3. **Card details entered incorrectly:** Typos in card number, expiry, or CVV
4. **Card not enabled for online transactions:** Some debit cards are only set up for ATM use
5. **International card not enabled:** For cards issued outside India, international online payments may be blocked by your bank

**Fixes:**

- Double-check card number, expiry month/year, and CVV (3-digit code on back for Visa/Mastercard; 4-digit code on front for Amex)
- Check your account balance and daily limit via your bank app or SMS
- Call your bank's customer care to enable online transactions on your card
- Try a different payment method (UPI, another card, ShopEase Wallet, Net Banking)

---

## 3D Secure (OTP) Issues

### OTP not received for card payment

3D Secure is a security step that requires you to verify your card payment via OTP sent to your registered mobile number.

**If OTP is not received:**
1. Wait 2–3 minutes before requesting a new OTP (sometimes delayed due to telecom congestion)
2. Check if your mobile number is correctly registered with your bank
3. Ensure you have network coverage (OTPs require an SMS-capable network)
4. Click **Resend OTP** on the payment page (usually available after 60 seconds)
5. Contact your bank to confirm the mobile number registered to your card

**OTP page timed out:**
- OTPs are typically valid for 5 minutes
- If it times out, click **Go Back** and restart the payment process
- Do not refresh the payment page — it can cause the transaction to get stuck

**Payment deducted but OTP page froze:**
- This is rare; the payment is usually not captured
- Check **My Orders** — if the order is not visible, the payment was not collected
- Any deducted amount will be automatically reversed within 5–7 business days

---

## UPI Payment Issues

### UPI transaction failed or pending

**"Payment Pending" status:**
- UPI payments sometimes show "pending" for up to 30 minutes due to bank processing
- Do NOT pay again — wait for the status to update
- If still pending after 30 minutes, check your UPI app for the transaction status

**UPI ID not found or invalid:**
- Verify the UPI ID you entered (no spaces, correct format like name@bank)
- Ask the recipient to share their UPI ID again

**"Exceeds transaction limit":**
- UPI has a per-transaction limit of Rs. 1 lakh (some banks allow up to Rs. 5 lakh with enhanced verification)
- Split large orders into smaller amounts (not recommended) or use Net Banking / Card for large orders

**UPI app crashing or not responding:**
- Update your UPI app to the latest version
- Clear cache: Settings → Apps → [UPI App] → Clear Cache
- Try a different UPI app (e.g., switch from PhonePe to Google Pay)

---

## PayPal Issues

### Cannot link PayPal account to ShopEase

**Steps to link PayPal:**
1. Go to **Account Settings** → **Saved Payment Methods** → **Add PayPal**
2. You will be redirected to PayPal to log in and authorise ShopEase
3. Once authorised, your PayPal account is linked

**Common issues:**
- **PayPal account not verified:** PayPal requires email verification and a linked bank account or card. Complete PayPal verification first.
- **PayPal currency mismatch:** ShopEase processes in INR. PayPal's currency conversion applies.
- **PayPal purchase protection:** PayPal may flag purchases from new merchants — approve the transaction in your PayPal app
- **VPN blocking:** Some VPN configurations block PayPal authentication. Disable VPN during the linking process.

---

## Bank Hold / Amount Reserved

### Why is money reserved (on hold) but not fully deducted?

Some banks place a temporary "authorisation hold" on funds during payment processing. This hold reserves the amount to ensure availability but is not a final deduction.

**If the order was placed successfully:** The hold will convert to a full deduction within 24–48 hours. Any hold above the order amount will be released automatically.

**If the order failed but money is on hold:** The hold will be automatically released within 5–7 business days (timeline varies by bank). You do not need to take any action. If the hold persists beyond 7 days, contact your bank with the ShopEase order attempt reference number (visible in your email).

---

## Currency Conversion Issues (International Payments)

For international customers or cards issued outside India:

- ShopEase charges in **Indian Rupees (INR)**
- Your bank applies the foreign exchange rate at the time of the transaction
- An international transaction fee may apply (typically 1–3.5% of the transaction value, levied by your bank — not ShopEase)
- If your card is not enabled for foreign currency transactions, the payment will be declined — contact your bank to enable this

---

## What to Do If Nothing Works

If your payment keeps failing after trying the above steps:

1. **Try a completely different payment method** (e.g., switch from card to UPI or Net Banking)
2. **Add money to your ShopEase Wallet** via Net Banking and pay with the wallet (most reliable method)
3. **Contact your bank's customer care** — they can see the exact decline reason on their end
4. **Contact ShopEase support** with:
   - Your order attempt details (time, amount, payment method)
   - The error message you received (screenshot if possible)
   - Your bank transaction reference number (if the amount was debited)

**ShopEase Support:**
- Email: support@shopease.com
- Phone: 1-800-SHOP-EASE (Monday–Friday, 9am–6pm)
- Live Chat: shopease.com (Monday–Saturday, 9am–8pm)
