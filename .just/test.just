test *args:
   uv run --frozen pytest {{args}}

health:
   just test tests/smoke_test.py
