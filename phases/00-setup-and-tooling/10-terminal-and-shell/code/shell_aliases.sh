#!/usr/bin/env bash
#
# الأسماء المستعارة لـ Shell ووظائفها لتطوير AI.
# مصدر هذا من ~/.bashrc أو ~/.zshrc:
# المصدر /path/to/shell_aliases.sh

# --- GPU ---

alias gpu='nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
alias gpuwatch='watch -n1 nvidia-smi'
alias gpumem='nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'
alias gpuprocs='nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv'

# --- مراقبة التدريب ---

alias killtraining='pkill -f "python.*train"'

killtrain() {
    if [ -z "$1" ]; then
        pkill -f "python.*train"
        echo "قتل جميع عمليات تدريب بايثون"    else
        pkill -f "$1"
        echo "مطابقة العمليات المقتولة: $1"    fi
}

# --- البيئات الافتراضية ---

alias ae='source .venv/bin/activate'
alias de='deactivate'
alias mkvenv='python -m venv .venv && source .venv/bin/activate'
alias uvvenv='uv venv && source .venv/bin/activate'

# --- سجل المشاهدة ---

alias watchloss='tail -f logs/*.log | grep --line-buffered "loss"'
alias watchacc='tail -f logs/*.log | grep --line-buffered "accuracy\|acc"'
alias watcherr='tail -f logs/*.log | grep --line-buffered "ERROR\|error\|Exception"'

taillog() {
    local pattern="${1:-loss}"
    tail -f logs/*.log 2>/dev/null | grep --line-buffered "$pattern"
}

# --- مساحة القرص (بيانات التدريب تملأ الأقراص بسرعة) ---

alias diskuse='df -h .'
alias bigfiles='find . -type f -size +100M | xargs du -h 2>/dev/null | sort -rh | head -20'
alias bigmodels='find . \( -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" -o -name "*.ckpt" -o -name "*.bin" \) | xargs du -h 2>/dev/null | sort -rh | head -20'

# --- اختبارات البيئة السريعة ---

alias checkgpu='python -c "import torch; print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"Device: {torch.cuda.get_device_name(0)}\") if torch.cuda.is_available() else None"'
alias checkcuda='env | grep -i cuda'
alias checkenv='python --version && pip --version && python -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}\")" 2>/dev/null'

# --- اختصارات tmux ---

alias ta='tmux attach -t'
alias tls='tmux ls'
alias tn='tmux new -s'
alias tk='tmux kill-session -t'

trainenv() {
    local name="${1:-train}"
    tmux new-session -d -s "$name"
    tmux split-window -h -t "$name"
    tmux split-window -v -t "$name"
    tmux send-keys -t "$name:0.1" 'watch -n1 nvidia-smi' C-m
    tmux send-keys -t "$name:0.2" 'htop' C-m
    tmux select-pane -t "$name:0.0"
    tmux attach -t "$name"
}

# --- SSH المساعدون ---

syncto() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "الاستخدام: المزامنة <المضيف> <remote_path> [local_path]"        echo "مثال: syncto gpu ~/data./data"        return 1
    fi
    local host="$1"
    local remote="$2"
    local local_path="${3:-.}"
    rsync -avz --progress "$local_path" "${host}:${remote}"
}

syncfrom() {
    if [ -z "$1" ] || [ -z "$2" ]; then
        echo "الاستخدام: syncfrom <host> <remote_path> [local_path]"        echo "مثال: syncfrom gpu ~/results./results"        return 1
    fi
    local host="$1"
    local remote="$2"
    local local_path="${3:-.}"
    rsync -avz --progress "${host}:${remote}" "$local_path"
}

# --- إدارة التجارب ---

newexp() {
    local name="${1:-experiment}"
    local dir="experiments/${name}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$dir/logs" "$dir/checkpoints" "$dir/configs"
    echo "دليل التجربة الذي تم إنشاؤه: $dir"    echo "$dir"
}

lastexp() {
    ls -dt experiments/*/ 2>/dev/null | head -1
}

# --- مساعدات تنزيل النماذج ---

hfdownload() {
    if [ -z "$1" ]; then
        echo "الاستخدام: hfdownload <model_id> [اسم الملف]"        echo "مثال: hfdownload meta-llama/Llama-2-7b config.json"        return 1
    fi
    local model="$1"
    local file="${2:-}"
    if [ -n "$file" ]; then
        wget "https://huggingface.co/${model}/resolve/main/${file}"
    else
        echo "استنساخ الريبو الكامل (استخدم git-lfs)..."        git lfs install
        git clone "https://huggingface.co/${model}"
    fi
}

# --- إدارة العمليات ---

memhogs() {
    ps aux --sort=-%mem 2>/dev/null | head -11 || ps aux -m | head -11
}

psg() {
    ps aux | grep -v grep | grep -i "$1"
}
