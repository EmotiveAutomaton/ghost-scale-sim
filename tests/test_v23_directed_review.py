from ghostscale.validation.soundingline.v23.directed_review import producer_signature

def test_review_repair_allows_checker_change_but_rejects_producer_change(tmp_path):
    original=tmp_path/'original.py';checker=tmp_path/'checker.py';producer=tmp_path/'producer.py'
    original.write_text('def run(): return 1\ndef reconstruct(): return 1\n')
    checker.write_text('def run(): return 1\ndef reconstruct(): return 2\n')
    producer.write_text('def run(): return 2\ndef reconstruct(): return 1\n')
    assert producer_signature(original)==producer_signature(checker)
    assert producer_signature(original)!=producer_signature(producer)
