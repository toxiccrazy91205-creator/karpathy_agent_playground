import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings
import os

from .models import PlaygroundSession
from .nvidia_nim import call_nvidia_nim, DEFAULT_NIM_BASE_URL, DEFAULT_NIM_MODEL
from openai import OpenAI

logger = logging.getLogger(__name__)

# List of popular NVIDIA NIM models to select from
POPULAR_MODELS = [
    {"id": "meta/llama-3.1-70b-instruct", "name": "Llama 3.1 70B Instruct (Default)"},
    {"id": "meta/llama-3.1-8b-instruct", "name": "Llama 3.1 8B Instruct"},
    {"id": "meta/llama3-70b-instruct", "name": "Llama 3 70B Instruct"},
    {"id": "mistralai/mixtral-8x22b-instruct-v0.1", "name": "Mixtral 8x22B Instruct"},
    {"id": "nvidia/llama-3.1-nemotron-51b-instruct", "name": "Llama 3.1 Nemotron 51B Instruct"},
]

def get_api_credentials(request):
    """
    Helper to fetch NVIDIA NIM credentials from session, fallback to env variables.
    """
    api_key = request.session.get('NVIDIA_API_KEY') or os.getenv('NVIDIA_API_KEY')
    base_url = request.session.get('NVIDIA_BASE_URL') or os.getenv('NVIDIA_BASE_URL') or DEFAULT_NIM_BASE_URL
    selected_model = request.session.get('NVIDIA_MODEL') or os.getenv('NVIDIA_MODEL') or DEFAULT_NIM_MODEL
    
    return api_key, base_url, selected_model

def dashboard(request):
    """
    Main dashboard page.
    """
    api_key, _, selected_model = get_api_credentials(request)
    recent_sessions = PlaygroundSession.objects.all()[:5]
    
    context = {
        'api_key_configured': bool(api_key),
        'selected_model': selected_model,
        'models': POPULAR_MODELS,
        'recent_sessions': recent_sessions,
    }
    return render(request, 'playground/dashboard.html', context)

def configure(request):
    """
    Configuration page for NVIDIA NIM API.
    """
    if request.method == 'POST':
        action = request.POST.get('action')
        
        # Capture form inputs
        api_key = request.POST.get('api_key', '').strip()
        base_url = request.POST.get('base_url', '').strip() or DEFAULT_NIM_BASE_URL
        model = request.POST.get('model', '').strip() or DEFAULT_NIM_MODEL
        
        if action == 'test_connection':
            # Run test connection
            if not api_key:
                return JsonResponse({'success': False, 'message': 'API Key is required to test connection.'})
            
            try:
                client = OpenAI(base_url=base_url, api_key=api_key)
                # Call low-token chat completions
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5
                )
                return JsonResponse({'success': True, 'message': 'Successfully connected to NVIDIA NIM API!'})
            except Exception as e:
                return JsonResponse({'success': False, 'message': f'Connection failed: {str(e)}'})
                
        elif action == 'save_config':
            # Save to session
            request.session['NVIDIA_API_KEY'] = api_key
            request.session['NVIDIA_BASE_URL'] = base_url
            request.session['NVIDIA_MODEL'] = model
            return redirect('dashboard')
            
    # GET request: load current config
    session_key = request.session.get('NVIDIA_API_KEY', '')
    env_key = os.getenv('NVIDIA_API_KEY', '')
    
    context = {
        'api_key': session_key or (f"******** (Loaded from Environment)" if env_key else ""),
        'base_url': request.session.get('NVIDIA_BASE_URL') or os.getenv('NVIDIA_BASE_URL') or DEFAULT_NIM_BASE_URL,
        'selected_model': request.session.get('NVIDIA_MODEL') or os.getenv('NVIDIA_MODEL') or DEFAULT_NIM_MODEL,
        'models': POPULAR_MODELS,
        'env_key_present': bool(env_key),
    }
    return render(request, 'playground/configure.html', context)

@require_POST
def execute_agent(request):
    """
    AJAX endpoint to run the Karpathy Agent prompt loop.
    """
    api_key, base_url, model = get_api_credentials(request)
    
    if not api_key:
        return JsonResponse({
            'success': False,
            'message': 'NVIDIA API Key not configured. Please go to settings and enter your credentials.'
        }, status=400)
        
    try:
        data = json.loads(request.body)
        original_code = data.get('original_code', '').strip()
        task_description = data.get('task_description', '').strip()
        custom_model = data.get('model', '').strip()
        
        if not original_code:
            return JsonResponse({'success': False, 'message': 'Original Code is required.'}, status=400)
        if not task_description:
            return JsonResponse({'success': False, 'message': 'Task Description is required.'}, status=400)
            
        model_to_use = custom_model if custom_model else model
        
        # Execute NVIDIA NIM call
        result = call_nvidia_nim(
            api_key=api_key,
            model=model_to_use,
            original_code=original_code,
            task_description=task_description,
            base_url=base_url
        )
        
        # Save to history db
        session = PlaygroundSession.objects.create(
            original_code=original_code,
            task_description=task_description,
            model_used=model_to_use,
            assumptions=result.get('think_before_coding', {}),
            simplicity_check=result.get('simplicity_check', {}),
            modified_code=result.get('modified_code', ''),
            verifiable_goals=result.get('verifiable_goals', []),
            raw_response=json.dumps(result)
        )
        
        return JsonResponse({
            'success': True,
            'session_id': session.id,
            'assumptions': session.assumptions,
            'simplicity_check': session.simplicity_check,
            'modified_code': session.modified_code,
            'verifiable_goals': session.verifiable_goals
        })
        
    except Exception as e:
        logger.exception("Error executing NIM call")
        return JsonResponse({
            'success': False,
            'message': f"Error calling NVIDIA NIM API: {str(e)}"
        }, status=500)

def history_list(request):
    """
    Lists past playground runs.
    """
    sessions = PlaygroundSession.objects.all()
    return render(request, 'playground/history_list.html', {'sessions': sessions})

def history_detail(request, pk):
    """
    Shows detail for a single past run.
    """
    session = get_object_or_404(PlaygroundSession, pk=pk)
    return render(request, 'playground/history_detail.html', {'session': session})
