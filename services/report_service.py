import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class DrivingSessionReport:
    start_time: datetime
    end_time: datetime | None = None
    total_driving_time_sec: float = 0.0
    total_yawns: int = 0
    total_microsleeps: int = 0
    total_phone_usages: int = 0
    eyes_off_road_sec: float = 0.0
    driving_safety_score: int = 100

class ReportService:
    def __init__(self):
        self.current_report: DrivingSessionReport | None = None
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

        # Temporary tracking variables
        self._last_update_time = None

    def start_session(self):
        now = datetime.now()
        self.current_report = DrivingSessionReport(start_time=now)
        self._last_update_time = now
        logger.info("Telemetri oturumu başlatıldı.")

    def update_telemetry(self, status: str, is_gaze_off_road: bool):
        if not self.current_report or not self._last_update_time:
            return

        now = datetime.now()
        delta_sec = (now - self._last_update_time).total_seconds()
        self._last_update_time = now

        self.current_report.total_driving_time_sec += delta_sec

        if is_gaze_off_road:
            self.current_report.eyes_off_road_sec += delta_sec

    def record_violation(self, violation_type: str):
        if not self.current_report:
            return

        if violation_type == "YAWNING":
            self.current_report.total_yawns += 1
        elif violation_type == "DROWSY":
            self.current_report.total_microsleeps += 1
        elif violation_type == "PHONE_USAGE":
            self.current_report.total_phone_usages += 1

    def end_session(self) -> tuple[DrivingSessionReport | None, Path | None]:
        if not self.current_report:
            return None, None

        self.current_report.end_time = datetime.now()
        
        # Sürüş Puanı Hesaplama (100 üzerinden)
        score = 100.0
        score -= self.current_report.total_yawns * 2.0
        score -= self.current_report.total_microsleeps * 10.0
        score -= self.current_report.total_phone_usages * 15.0
        
        # Gözlerin yoldan ayrılma süresine göre ceza (her saniye için 0.5 puan)
        score -= self.current_report.eyes_off_road_sec * 0.5

        self.current_report.driving_safety_score = max(0, int(round(score)))

        self._save_report()
        report = self.current_report
        pdf_path = getattr(self, 'pdf_path', None)
        self.current_report = None
        self._last_update_time = None
        return report, pdf_path

    def _save_report(self):
        if not self.current_report:
            return

        timestamp = self.current_report.start_time.strftime("%Y%m%d_%H%M%S")
        self.pdf_path = self.reports_dir / f"Surus_Raporu_{timestamp}.pdf"
        self._generate_pdf(self.pdf_path, self.current_report)
        logger.info(f"Sürüş raporu PDF olarak kaydedildi: {self.pdf_path}")

    def _generate_pdf(self, path: Path, report: DrivingSessionReport):
        import os
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        # Windows varsayılan Arial fontunu kaydet
        font_path = "C:/Windows/Fonts/arial.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("Arial-TR", font_path))
            default_font = "Arial-TR"
        else:
            default_font = "Helvetica"

        doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        styles = getSampleStyleSheet()
        elements = []

        # Update default font for styles
        styles["Normal"].fontName = default_font
        styles["Heading1"].fontName = default_font
        styles["Heading3"].fontName = default_font
        styles["Title"].fontName = default_font

        # Styles
        title_style = ParagraphStyle(
            name="TitleStyle",
            parent=styles["Title"],
            fontSize=18,
            fontName=default_font,
            textColor=colors.HexColor("#1976D2"),
            spaceAfter=20
        )
        
        score_style = ParagraphStyle(
            name="ScoreStyle",
            parent=styles["Heading1"],
            fontSize=24,
            fontName=default_font,
            textColor=colors.HexColor("#388E3C") if report.driving_safety_score > 70 else colors.HexColor("#D32F2F"),
            alignment=1,
            spaceAfter=20
        )

        normal_style = styles["Normal"]
        
        # 1. Başlık
        elements.append(Paragraph("<b>Akıllı Sürücü Güvenlik ve Dikkat Takip Sistemi - Sefer Analiz Raporu</b>", title_style))
        
        # 2. Meta Veriler
        start_str = report.start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_str = report.end_time.strftime("%Y-%m-%d %H:%M:%S") if report.end_time else "Bilinmiyor"
        minutes = int(report.total_driving_time_sec // 60)
        seconds = int(report.total_driving_time_sec % 60)
        
        meta_data = (
            f"<b>Oturum Başlangıcı:</b> {start_str}<br/>"
            f"<b>Oturum Bitişi:</b> {end_str}<br/>"
            f"<b>Toplam Sürüş Süresi:</b> {minutes} dakika {seconds} saniye<br/>"
        )
        elements.append(Paragraph(meta_data, normal_style))
        elements.append(Spacer(1, 20))
        
        # 3. Genel Değerlendirme Skoru
        elements.append(Paragraph(f"<b>Sürüş Güvenlik Skoru: {report.driving_safety_score} / 100</b>", score_style))
        elements.append(Spacer(1, 20))
        
        # 4. İhlal Tablosu
        data = [
            ["İhlal Türü", "Detay / Sayı"],
            ["Mikro Uyku / Göz Kapama", f"{report.total_microsleeps} kez"],
            ["Esneme", f"{report.total_yawns} kez"],
            ["Telefon Kullanımı", f"{report.total_phone_usages} kez"],
            ["Gözlerin Yoldan Ayrılma Süresi", f"{report.eyes_off_road_sec:.1f} saniye"]
        ]
        
        table = Table(data, colWidths=[250, 200])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1976D2")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), default_font),
            ('FONTNAME', (0, 1), (-1, -1), default_font),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E3F2FD")),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#BBDEFB"))
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 30))
        
        # 5. Yapay Zeka Tavsiyesi
        if report.driving_safety_score > 85:
            advice = "Harika! Sürüşünüz son derece güvenli ve dikkatli tamamlandı. Bu dikkatinizi korumaya devam edin."
            advice_color = "#388E3C"
        elif report.driving_safety_score >= 70:
            advice = "Sürüşünüz genel olarak iyiydi ancak ufak tefek dikkat dağınıklıkları tespit edildi. Lütfen yola daha fazla odaklanın."
            advice_color = "#F57C00"
        else:
            advice = "DİKKAT: Kritik seviyede yorgunluk ve dikkat kaybı tespit edildi! Lütfen en kısa sürede güvenli bir yere park edip mola verin."
            advice_color = "#D32F2F"
            
        advice_style = ParagraphStyle(
            name="AdviceStyle",
            parent=normal_style,
            fontSize=12,
            textColor=colors.HexColor(advice_color),
            spaceBefore=10
        )
        
        elements.append(Paragraph("<b>Sistem Tavsiyesi:</b>", styles["Heading3"]))
        elements.append(Paragraph(advice, advice_style))
        
        doc.build(elements)
