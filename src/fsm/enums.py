from enum import Enum


class Language(str, Enum):
    ES = "es"
    EN = "en"


class Intent(str, Enum):
    ANSWER = "answer"
    QUESTION = "question"
    CHITCHAT = "chitchat"
    STOP = "stop"
    UNCLEAR = "unclear"


class Sentiment(str, Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    FRUSTRATED = "frustrated"
    CONFUSED = "confused"


class Availability(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    WEEKENDS = "weekends"


class Schedule(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    FLEXIBLE = "flexible"


class Modality(str, Enum):
    TEXT = "text"
    VOICE = "voice"


class Stage(str, Enum):
    CONSENT = "consent"
    NAME = "name"
    LICENSE = "license"
    CITY = "city"
    AVAILABILITY = "availability"
    SCHEDULE = "schedule"
    EXPERIENCE = "experience"
    START_DATE = "start_date"
    SUMMARY = "summary"


class Action(str, Enum):
    ASK = "ask"
    CLARIFY = "clarify"
    ANSWER_QUESTION = "answer_question"
    CONFIRM_SUMMARY = "confirm_summary"
    CLOSE_QUALIFIED = "close_qualified"
    CLOSE_DISQUALIFIED_NO_LICENSE = "close_disqualified_no_license"
    CLOSE_OUT_OF_AREA = "close_out_of_area"
    CLOSE_CONSENT_DECLINED = "close_consent_declined"
    CLOSE_OPTED_OUT = "close_opted_out"


class Outcome(str, Enum):
    IN_PROGRESS = "in_progress"
    QUALIFIED = "qualified"
    DISQUALIFIED_NO_LICENSE = "disqualified_no_license"
    OUT_OF_AREA = "out_of_area"
    CONSENT_DECLINED = "consent_declined"
    OPTED_OUT = "opted_out"
    ABANDONED = "abandoned"


TERMINAL_OUTCOMES = frozenset(
    o for o in Outcome if o is not Outcome.IN_PROGRESS
)
