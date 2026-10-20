from datetime import datetime
import os
import json
import io

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Codespaces has no screen, so charts are saved as PNG files
import matplotlib.pyplot as plt
from dotenv import load_dotenv  
from groq import Groq, APIError, RateLimitError  

load_dotenv()


class DataAnalystAgent:
    """
    وكيل ذكي متخصص في تحليل البيانات
    """

    def __init__(self,
                 model: str = "openai/gpt-oss-120b",  
                 max_history: int = 10):

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it as a Codespaces secret and restart the codespace."
            )
        self.client = Groq(api_key=api_key)

        self.model = model
        self.history = []
        self.max_history = max_history
        self.df = None  # DataFrame لتخزين البيانات المحملة
        self.df_info = None  # لتخزين معلومات DataFrame
        self.chart_count = 0

         # تعليمات الوكيل المتخصصة في تحليل البيانات
        self.system_prompt = """
        أنت وكيل ذكاء اصطناعي متخصص في تحليل البيانات.
        مهامك هي:
        1. تحليل الداتا وفهم الأعمدة والقيم
        2. كتابة كود باندا مناسب لكل طلب
        3. تفسير النتائج بطريقة واضحة وبسيطة
        4. اقتراح تحليلات مفيدة
         لما تكتب كود دايما حطه بين ```python و ```
        الداتا محملة بالفعل في متغير اسمه df، متقراش الملف تاني.
        خزن الناتج النهائي في متغير اسمه result.
        للرسم استخدم matplotlib.pyplot as plt ومتستخدمش plt.show().
        """

    def load_data(self, source, file_type: str):
        """تحميل البيانات من ملف"""
        try:
            # accept a URL or a file path as well as raw bytes
            if isinstance(source, bytes):
                source = io.BytesIO(source)
            elif isinstance(source, str) and "github.com" in source and "/blob/" in source:
                # a github.com/.../blob/... link is an HTML page; the raw link is the actual file
                source = source.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

            if file_type == "csv":
                self.df = pd.read_csv(source)
            elif file_type == "excel":
                self.df = pd.read_excel(source)
            else:
                raise ValueError("نوع الملف غير مدعوم. استخدم csv أو excel.")

            # حفظ معلومات DataFrame
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            self.df_info = f"""
            {buffer.getvalue()}


            إحصائيات إضافية:
            القيم الفاضية: {self.df.isnull().sum().to_dict()}
            إحصائيات سريعة: {self.df.describe().to_string()}
            """
            return f"تم تحميل الداتا بنجاح ,عندنا {self.df.shape[0]} صف و {self.df.shape[1]} عمود"

        except Exception as e:
            return f"خطأ في تحميل الداتا: {str(e)}"

    def get_context(self) -> list:
        """بناء السياق من السجل"""
        system = self.system_prompt
        # لو في داتا محملة بيضيف معلوماتها للسياق
        if self.df_info:
            system += f"\n[معلومات الداتا المحملة]\n{self.df_info}"

        messages = [{"role": "system", "content": system}]
        for msg in self.history[-5:]:  # آخر 5 رسائل فقط
            messages.append({"role": msg["role"], "content": msg["content"]})
        return messages

    def execute_code(self, response: str) -> str:
        """
        استخراج وتنفيذ الكود من رد الموديل
        بيدور على الكود جوه الرد وبينفذه على الداتا الحقيقية
        """
        if "```python" not in response:
            return response

        try:
            # استخراج الكود من الرد
            code = response.split("```python")[1].split("```")[0].strip()

            # تنفيذ الكود مع الداتا المحملة
            plt.close("all")
            local_vars = {"df": self.df, "pd": pd, "plt": plt}
            exec(code, local_vars)

            # the code drew a chart, save it so we can open it in Codespaces
            if plt.get_fignums():
                self.chart_count += 1
                chart_file = f"chart_{self.chart_count}.png"
                plt.savefig(chart_file, bbox_inches="tight")
                response += f"\n\nتم حفظ الرسم في {chart_file}"

            if "result" in local_vars:
                return f"{response}\n\nالنتيجة:\n{local_vars['result']}"

            return response

        except Exception as e:
            return f"{response}\n\nخطأ في تنفيذ الكود: {str(e)}"

    def add_to_history(self, role: str, content: str):
        """حفظ الرسائل مع التحكم في العدد"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def export_session(self, filename: str = "analysis_session.json"):
        """تصدير الجلسة الحالية لسجل JSON"""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "conversation": self.history,
            "data_info": self.df_info
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(session_data, f, ensure_ascii=False, indent=4)
        print(f"تم حفظ الجلسة في {filename}")

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        إرسال الطلب للموديل
        التمبريتشر 0.3 عشان الردود تكون دقيقة مش عشوائية
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.get_context(),
                temperature=temperature,
            )
            return completion.choices[0].message.content.strip()

        except RateLimitError:
            return "خطأ: تخطيت حد الطلبات المجاني في Groq، استنى دقيقة وحاول تاني"
        except APIError as e:
            return f"خطأ في الاتصال بـ Groq: {str(e)}"
        except Exception as e:
            return f"خطأ: {str(e)}"

    def chat(self, user_input: str) -> str:
        """المحادثة الكاملة مع تنفيذ الكود"""
        self.add_to_history("user", user_input)

        # جيب الرد من الموديل
        response = self.generate(user_input)

        # لو في داتا محملة نفذ الكود الموجود في الرد
        if self.df is not None:
            response = self.execute_code(response)

        self.add_to_history("assistant", response)
        return response

# ================== طريقة الاستخدام ==================

if __name__ == "__main__":

    agent = DataAnalystAgent()
    url = "https://github.com/ayaelsadek/Agentic-AI_Deeplearning/blob/main/train.csv"
    print(agent.load_data(url, "csv"))  

    questions = [
        "كام صف فاضي في الداتا؟",
        "اعملي تحليل إحصائي سريع للأعمدة الرقمية",
        "ايه العمود اللي فيه أكتر قيم فاضية؟",
        "اعملي رسم بياني للتوزيع"
    ]

    for question in questions:
        print(f"\nالسؤال: {question}")
        print(f"الرد: {agent.chat(question)}")

    # حفظ الجلسة
    agent.export_session()
