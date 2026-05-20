from django.db import models

class PlaygroundSession(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    original_code = models.TextField()
    task_description = models.TextField()
    model_used = models.CharField(max_length=150)
    
    # Store LLM structured responses as JSON
    # Django 4.0+ natively supports JSONField for SQLite out-of-the-box
    assumptions = models.JSONField(default=dict)
    simplicity_check = models.JSONField(default=dict)
    modified_code = models.TextField()
    verifiable_goals = models.JSONField(default=list)
    
    raw_response = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Session {self.id} - {self.model_used} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
