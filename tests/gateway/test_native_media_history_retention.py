"""Transcript-owned native input survives settlement and age in either path spelling.

Explicit retirement of its owned session, not age alone, permits collection.
"""
import os
import time

import pytest

from gateway.session_ingress_media import _media_root, capture_native_media, release_admission_media
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input, settle_session_input


def _settled_native_document(tmp_path, monkeypatch, note):
    from gateway.platforms.base import get_document_cache_dir
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    staged = get_document_cache_dir() / 'doc_1_report.pdf'
    staged.write_bytes(b'%PDF-1.4 retained')
    media = capture_native_media([staged])
    retained = media[0]['path']
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='telegram')
    db.append_message('s', 'user', note(retained))
    epoch = begin_runtime_epoch(db, instance_id='owner')
    row = admit_session_input(db, epoch=epoch, principal_id='p', session_id='s', request_id='m1',
                              payload={'text': 'page 3?', 'native_text_v1': {'media': media}})
    claim = claim_session_input(db, epoch=epoch, session_id='s')
    settle_session_input(db, epoch=epoch, admission_id=row['admission_id'], generation=claim['generation'],
                         outcome='completed')
    return db, row, retained


@pytest.mark.parametrize('note', [
    lambda path: f"[The user sent a document: 'report.pdf'. It is saved at: {path}.]",
    # A docker/modal backend's agent-visible spelling of the same retained file.
    lambda path: "It is saved at: /root/.hermes/cache/documents/" + path.split('/documents/', 1)[1],
], ids=['host-path', 'agent-visible-path'])
def test_settled_native_document_named_by_the_transcript_survives_settlement(tmp_path, monkeypatch, note):
    import gateway.session_ingress_media as media_module
    monkeypatch.setattr(media_module, '_last_sweep', {}, raising=False)
    db, row, retained = _settled_native_document(tmp_path, monkeypatch, note)
    try:
        release_admission_media(db, row['admission_id'])
        assert os.path.exists(retained), 'the follow-up turn is told to open a path settlement deleted'
        # A quota wait can outlast the cache age; a follow-up still needs these bytes.
        old = time.time() - 25 * 3600
        os.utime(retained, (old, old))
        monkeypatch.setattr(media_module, '_last_sweep', {}, raising=False)
        release_admission_media(db, row['admission_id'])
        assert os.path.exists(retained), 'age erased an attachment still owned by history'
        # This fresh isolated test session is explicitly discarded, releasing its owners.
        assert db.delete_session('s')
        assert not os.path.exists(retained) and list(_media_root().iterdir()) == []
    finally:
        db.close()
