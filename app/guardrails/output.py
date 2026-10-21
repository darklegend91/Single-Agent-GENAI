"""The model selects claims; this module alone renders verified source facts."""
import re
from dataclasses import dataclass, field
from typing import Any

from app.data.dictionary import FIELD_DICTIONARY, MONTHLY_FIELDS
from app.errors import ValidationFailure
from app.guardrails.input import INJECTION
from app.guardrails.policy import LIMITATIONS
from app.schemas import AgentResult, CustomerAnswer, Evidence
from app.services.analytics import format_rupees


@dataclass
class EvidenceLedger:
    """Server-created, request-local evidence; never accept a ledger from clients."""
    user_id: str
    values: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    metric_fields: dict[str, tuple[str, ...]] = field(default_factory=dict)
    limitations: set[str] = field(default_factory=set)

    def add_domain(self, result):
        if result.user_id != self.user_id:
            raise ValidationFailure()
        self.values.update({key:value for key,value in result.values.items() if key in result.provenance})
        self.limitations.update(result.limitations)

    def add_revenue(self, result):
        if result.user_id != self.user_id:
            raise ValidationFailure()
        self.metrics.update(monthly_settled_total_paise=result.total_paise, monthly_settled_complete=result.complete, monthly_settled_available_months=len(result.available_months))
        for key, provenance in result.metric_provenance.items():
            if key not in METRIC_UNITS or set(provenance.source_fields)-set(MONTHLY_FIELDS):
                raise ValidationFailure()
            # Absent schema columns must never appear as existing source citations.
            self.metric_fields[key]=tuple(name for name in provenance.source_fields if name in result.provenance)
        self.limitations.update(result.limitations)
        for month,value in result.monthly_paise.items():
            name=f'SettledRevenue{month}Paise'
            if name in result.provenance:
                self.values[name]=value

METRIC_UNITS = {'monthly_settled_total_paise':'paise','monthly_settled_complete':'boolean','monthly_settled_available_months':'count'}
METRIC_LABELS = {'monthly_settled_total_paise':'Monthly settled total (available months)','monthly_settled_complete':'All twelve monthly settled values available','monthly_settled_available_months':'Available monthly settled values'}

def controlled_answer(status='insufficient_evidence', code='no_relevant_evidence'):
    reason=LIMITATIONS[code]
    answer=('I cannot assist with that request. '+reason if status=='refused' else 'I could not answer from the available source data. '+reason)
    return CustomerAnswer(status=status, answer=answer, limitations=[reason])

def _format(value, unit):
    if isinstance(value,str):
        if not value.strip() or len(value)>500 or INJECTION.search(value) or any(ord(c)<32 for c in value):
            raise ValidationFailure()
        return value
    if unit=='paise':
        return f'{value} paise (INR {format_rupees(value)})'
    if type(value) is bool:
        return 'true' if value else 'false'
    return str(value)

def validate_result(result: AgentResult, context, tool_evidence: EvidenceLedger) -> CustomerAnswer:
    if result.status != 'answered':
        if result.claims:
            raise ValidationFailure()
        return controlled_answer(result.status, 'unsupported_question')
    if not result.claims:
        return controlled_answer()
    lines, evidence = [], []
    codes=set(tool_evidence.limitations) | set(context.limitations) | set(result.limitations)
    seen=set(); seen_citations=set()
    for claim in result.claims:
        if claim.user_id != tool_evidence.user_id:
            raise ValidationFailure()
        if claim.source_field:
            name=claim.source_field
            definition=FIELD_DICTIONARY.get(name)
            if definition is None or name not in tool_evidence.values:
                raise ValidationFailure()
            value=tool_evidence.values[name]
            expected_unit=definition.unit
            fields=[name]
            label=re.sub(r'(?<=[a-z])(?=[A-Z])',' ',name)
            if definition.kind=='code': codes.add('unknown_code')
            if definition.kind=='timestamp': codes.add('unresolved_timestamp')
        else:
            name=claim.metric_id
            value=tool_evidence.metrics.get(name)
            expected_unit=METRIC_UNITS[name]
            fields=list(tool_evidence.metric_fields.get(name,()))
            label=METRIC_LABELS[name]
            if not fields: raise ValidationFailure()
        if value is None or (isinstance(value,str) and not value.strip()) or type(claim.value) is not type(value) or claim.value != value or claim.unit != expected_unit:
            raise ValidationFailure()
        if name in seen: continue
        seen.add(name)
        lines.append(f'{label}: {_format(value,expected_unit)}.')
        citation=(tuple(fields),bool(claim.source_field))
        if citation not in seen_citations:
            seen_citations.add(citation)
            evidence.append(Evidence(user_id=tool_evidence.user_id,source_fields=fields,explanation='Verified against the authorized source record.' if claim.source_field else 'Verified deterministic calculation from the cited source fields.'))
    return CustomerAnswer(status='answered',answer='\n'.join(lines),evidence=evidence,limitations=[LIMITATIONS[code] for code in sorted(codes) if code in LIMITATIONS])
