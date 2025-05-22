import os
import uuid
import json
from quart import Quart, render_template, request, send_file, jsonify
from ai_pipeline import run_ai_pipeline
import aiofiles

app = Quart(__name__)
UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'reports'
POLICY_FOLDER = 'policies'
STATIC_FOLDER = 'static'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/upload', methods=['POST'])
async def upload():
    try:
        form = await request.form
        file = (await request.files)['model_card']
        filename = f"{uuid.uuid4()}_{file.filename}"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        await file.save(upload_path)

        # Get selected policies from form data
        selected_policies = None
        if 'selected_policies' in form:
            try:
                selected_policies = json.loads(form['selected_policies'])
            except json.JSONDecodeError:
                selected_policies = None

        # Read the model card content
        async with aiofiles.open(upload_path, 'r', encoding='utf-8') as f:
            model_card_content = await f.read()

        # Run AI pipeline with uploaded file and policies
        report_path = os.path.join(REPORT_FOLDER, f"{filename}_report.txt")
        heatmap_filenames, summaries, top_level_summary, section_summaries = await run_ai_pipeline(upload_path, POLICY_FOLDER, report_path, selected_policies)
        
        # Check if heatmap HTML files were generated
        if heatmap_filenames and all(os.path.exists(os.path.join(STATIC_FOLDER, filename)) for filename in heatmap_filenames):
            return jsonify({
                'success': True, 
                'heatmap_filenames': heatmap_filenames,
                'model_card_content': model_card_content,
                'summaries': summaries,
                'top_level_summary': top_level_summary,
                'section_summaries': section_summaries
            })
        else:
            return jsonify({'success': False, 'error': 'Heatmap generation failed'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/static/<path:filename>')
async def serve_static(filename):
    return await send_file(os.path.join(STATIC_FOLDER, filename))

if __name__ == '__main__':
    app.run(port=8000)