"""Chart helpers. class_charts.py / student_charts.py were corrupted in the old project."""

try:
    from visualization.student_charts import generate_student_charts
    from visualization.class_charts import generate_class_charts
except Exception:
    generate_student_charts = None
    generate_class_charts = None

__all__ = ["generate_student_charts", "generate_class_charts"]
