'use strict';

// The supplied parser, unchanged apart from its exported function name.
function supplied(llmOutput) {
    const normalized = String(llmOutput || '').trim().toUpperCase();
    if (!normalized) {
        return { blocked: true, parseOk: false, normalized: '' };
    }
    const match = normalized.match(/\b(BLOCK|OK|SAFE|ALLOW|PASS|APPROVE|APPROVED|CLEAN|YES|REJECT|DENY|UNSAFE|REMOVE|FILTER|NO)\b/);
    const label = match ? match[1] : '';
    if (!label) {
        return { blocked: true, parseOk: false, normalized };
    }
    if (label === 'BLOCK' || label === 'REJECT' || label === 'DENY' || label === 'UNSAFE' || label === 'REMOVE' || label === 'FILTER' || label === 'NO') {
        return { blocked: true, parseOk: true, normalized: label };
    }
    return { blocked: false, parseOk: true, normalized: label };
}

// Baseline: separate the effect of default blocking from tolerant parsing.
function exactWithFallback(output, allow) {
    const text = String(output || '').trim();
    return text === allow || text === 'BLOCK'
        ? {blocked: text === 'BLOCK', parseOk: true, normalized: text}
        : {blocked: true, parseOk: false, normalized: text};
}

// A bounded format parser, not another natural-language moderation model.
function enhanced(output) {
    let text = String(output || '').trim();
    const fail = () => ({blocked: true, parseOk: false, normalized: text});
    const fence = text.match(/^```(?:json|text)?\s*\n?([\s\S]*?)\n?```$/i);
    if (fence) text = fence[1].trim();
    // Only a single verdict field is accepted: duplicate/conflicting fields fail.
    const json = text.match(/^\{\s*"(?:label|decision|verdict)"\s*:\s*"([^"\\]*)"\s*\}$/i);
    if (json) text = json[1].trim();
    text = text.toUpperCase().replace(/[.!]+$/, '').trim();
    for (const mark of ['**', '__', '`', '"', "'"]) {
        if (text.startsWith(mark) && text.endsWith(mark) && text.length > mark.length * 2) {
            text = text.slice(mark.length, -mark.length).trim();
        }
    }
    text = text.replace(/^(?:DECISION|VERDICT|LABEL)\s*:\s*/, '')
        .replace(/^(?:THIS MESSAGE IS|THE MESSAGE IS|IT IS|THIS IS)\s+/, '')
        .replace(/[.!]+$/, '').trim();
    const allow = new Set(['OK', 'SAFE', 'ALLOW', 'ALLOWED', 'PASS', 'APPROVE', 'APPROVED', 'CLEAN']);
    const block = new Set(['BLOCK', 'BLOCKED', 'REJECT', 'REJECTED', 'DENY', 'DENIED', 'UNSAFE', 'REMOVE', 'FILTER']);
    let label = text;
    if (/^(?:NOT (?:OK|SAFE|CLEAN|ALLOWED)|DO NOT ALLOW|DON'T ALLOW)$/.test(text)) label = 'BLOCK';
    else if (/^(?:NOT UNSAFE|DO NOT BLOCK|DON'T BLOCK)$/.test(text)) label = 'OK';
    else if (!allow.has(label) && !block.has(label)) {
        // A leading verdict may be followed by a reason. Opposite labels or
        // uncertainty in that reason make this an unparsed, blocked response.
        const leading = text.match(/^(OK|SAFE|ALLOW|ALLOWED|PASS|APPROVE|APPROVED|CLEAN|BLOCK|BLOCKED|REJECT|REJECTED|DENY|DENIED|UNSAFE|REMOVE|FILTER)(?:\s*[:.!-]\s+|\s+BECAUSE\s+)([\s\S]+)$/);
        if (!leading) return fail();
        label = leading[1];
        const rest = leading[2];
        const words = rest.match(/\b(?:OK|SAFE|ALLOW|ALLOWED|PASS|APPROVE|APPROVED|CLEAN|BLOCK|BLOCKED|REJECT|REJECTED|DENY|DENIED|UNSAFE|REMOVE|FILTER)\b/g) || [];
        if (words.some(word => allow.has(word) !== allow.has(label)) || /\b(?:NOT|NEVER|MAYBE|UNSURE|UNCERTAIN|ACTUALLY|HOWEVER)\b/.test(rest)) return fail();
    }
    // YES and NO have no fixed meaning without knowing which question was asked.
    if (!allow.has(label) && !block.has(label)) return fail();
    const blocked = block.has(label);
    return {blocked, parseOk: true, normalized: blocked ? 'BLOCK' : 'OK'};
}

module.exports = {supplied, exactWithFallback, enhanced};
