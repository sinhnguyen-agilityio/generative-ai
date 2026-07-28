# models/enums.py

from enum import Enum


class UrgencyLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class EscalationTeam(str, Enum):
    NONE = "None"
    CUSTOMER_SUPPORT = "Customer Support"
    BILLING = "Billing"
    TECHNICAL_SUPPORT = "Technical Support"
    SHIPPING = "Shipping"


class SentimentLabel(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


class IntentCategory(str, Enum):
    ACCOUNT_SUPPORT = "Account Support"
    ORDER_STATUS = "Order Status"
    REFUND_REQUEST = "Refund Request"
    PAYMENT_ISSUE = "Payment Issue"
    SHIPPING = "Shipping"
    PRODUCT_QUESTION = "Product Question"
    TECHNICAL_ISSUE = "Technical Issue"
    COMPLAINT = "Complaint"
    FEATURE_REQUEST = "Feature Request"
    OTHER = "Other"
