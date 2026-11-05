"""
Aurora AX3000 Wi-Fi Router User Manual Generator
Uses ReportLab to programmatically generate a realistic, structured, 16-20 page
product user manual with headings, numbered steps, bullet points, tables,
error codes, and technical specifications.
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print 'Page X of Y' headers and footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Skip cover page
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#555555"))
        # Header
        self.drawString(54, 750, "Aurora AX3000 Wi-Fi 6 Router — User Manual")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 744, 558, 744)
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        self.drawString(54, 40, "Confidential & Proprietary — Aurora Networks Inc.")
        self.line(54, 52, 558, 52)
        self.restoreState()


def build_manual_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=64,
        bottomMargin=64
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Hierarchy
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor("#0B2545"),
        alignment=0
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#134074")
    )
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555")
    )
    h1_style = ParagraphStyle(
        'ManualH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0B2545"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'ManualH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#134074"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'ManualBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#222222"),
        spaceBefore=3,
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        'ManualBullet',
        parent=body_style,
        leftIndent=16,
        firstLineIndent=-10,
        spaceBefore=2,
        spaceAfter=2
    )
    step_style = ParagraphStyle(
        'ManualStep',
        parent=body_style,
        leftIndent=20,
        firstLineIndent=-14,
        spaceBefore=3,
        spaceAfter=3
    )
    warning_style = ParagraphStyle(
        'ManualWarning',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor("#8B0000"),
        spaceBefore=4,
        spaceAfter=4
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#222222")
    )
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    story = []

    # =========================================================================
    # PAGE 1: COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 100))
    story.append(Paragraph("Aurora AX3000 Wi-Fi 6 Router", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Dual-Band Gigabit Wireless Router • Model: AX3000-PRO", subtitle_style))
    story.append(Spacer(1, 25))
    story.append(HRFlowable(width="100%", thickness=3, color=colors.HexColor("#0B2545"), spaceAfter=30))
    story.append(Paragraph("Comprehensive User Guide & Hardware Reference Manual", subtitle_style))
    story.append(Spacer(1, 150))
    story.append(Paragraph("<b>Document Version:</b> 3.2.0<br/><b>Publication Date:</b> November 2026<br/><b>Firmware Compatibility:</b> v1.4.0 and above<br/><b>Hardware Revision:</b> Rev B1", meta_style))
    story.append(Spacer(1, 30))
    story.append(Paragraph("© 2026 Aurora Networks Inc. All rights reserved.", meta_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: TABLE OF CONTENTS & QUICK SPECS SUMMARY
    # =========================================================================
    story.append(Paragraph("Table of Contents", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC"), spaceAfter=12))
    
    toc_data = [
        ["1. Product Overview and Hardware Interfaces", "Page 3"],
        ["   1.1 Package Contents", "Page 3"],
        ["   1.2 Front Panel LED Status Indicators", "Page 3"],
        ["   1.3 Rear Panel Ports and Buttons", "Page 4"],
        ["2. Quick Start and Hardware Installation", "Page 5"],
        ["   2.1 Connecting the Hardware", "Page 5"],
        ["   2.2 Establishing Connection", "Page 5"],
        ["3. Web GUI and Initial Setup Wizard", "Page 6"],
        ["   3.1 Accessing the Web Management Console", "Page 6"],
        ["   3.2 Internet Connection Configuration", "Page 6"],
        ["   3.3 Administrator Account Protection", "Page 7"],
        ["4. Wireless Network Configuration", "Page 8"],
        ["   4.1 Dual-Band Wi-Fi Settings (2.4 GHz and 5 GHz)", "Page 8"],
        ["   4.2 Guest Network Setup and Isolation", "Page 9"],
        ["   4.3 Wi-Fi Security Protocols (WPA2 vs WPA3)", "Page 9"],
        ["5. Advanced Network Management", "Page 10"],
        ["   5.1 Quality of Service (QoS) Bandwidth Prioritization", "Page 10"],
        ["   5.2 Parental Controls and Domain Filtering", "Page 10"],
        ["   5.3 Port Forwarding, NAT and DMZ Rules", "Page 11"],
        ["   5.4 VPN Server (OpenVPN and WireGuard)", "Page 12"],
        ["   5.5 USB 3.0 Storage and Media Server", "Page 12"],
        ["6. System Maintenance and Recovery", "Page 13"],
        ["   6.1 Firmware Upgrade Procedures", "Page 13"],
        ["   6.2 Backup and Restoration of Configuration", "Page 13"],
        ["   6.3 Factory Reset and Reboot Methods", "Page 14"],
        ["7. Troubleshooting and Diagnostic Error Codes", "Page 15"],
        ["   7.1 Comprehensive Diagnostic Error Code Table", "Page 15"],
        ["   7.2 Diagnostic Procedures for Common Issues", "Page 16"],
        ["8. Technical Specifications and Hardware Architecture", "Page 17"],
        ["9. Regulatory Compliance, Safety and Warranty", "Page 18"]
    ]
    t_toc = Table([[Paragraph(c[0], body_style), Paragraph(c[1], body_style)] for c in toc_data], colWidths=[380, 100])
    t_toc.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('TOPPADDING', (0,0), (-1,-1), 1),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 1: PRODUCT OVERVIEW & HARDWARE INTERFACES
    # =========================================================================
    story.append(Paragraph("1. Product Overview and Hardware Interfaces", h1_style))
    story.append(Paragraph(
        "The Aurora AX3000 Wi-Fi 6 Router (Model: AX3000-PRO) delivers ultra-fast, reliable dual-band wireless "
        "connectivity designed for smart homes and demanding network environments. Powered by the 802.11ax standard, "
        "the router provides aggregate speeds up to 2976 Mbps (574 Mbps on 2.4 GHz and 2402 Mbps on 5 GHz) with "
        "160 MHz channel support, 1024-QAM, OFDMA, and MU-MIMO technology.", body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1.1 Package Contents", h2_style))
    story.append(Paragraph("Ensure the retail package contains all the following items before proceeding with setup:", body_style))
    story.append(Paragraph("• 1x Aurora AX3000 Wi-Fi 6 Router (AX3000-PRO)", bullet_style))
    story.append(Paragraph("• 1x Power Adapter (Input: 100-240V AC 50/60Hz, Output: 12 V DC, 2.0 A)", bullet_style))
    story.append(Paragraph("• 1x RJ-45 Cat 6 Gigabit Ethernet Cable (1.5 m / 5 ft)", bullet_style))
    story.append(Paragraph("• 1x Quick Installation Guide and Warranty Card", bullet_style))
    story.append(Paragraph("• 1x Wi-Fi Configuration Info Card with Default Network Credentials", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1.2 Front Panel LED Status Indicators", h2_style))
    story.append(Paragraph(
        "The front panel features six multi-color LED indicators that report real-time system, internet, and wireless "
        "status. Refer to the diagnostic table below to interpret LED behavior.", body_style
    ))
    story.append(Spacer(1, 4))

    led_table_data = [
        [Paragraph("LED Indicator", table_header), Paragraph("State", table_header), Paragraph("Color", table_header), Paragraph("Description & Diagnostic Meaning", table_header)],
        [Paragraph("POWER", table_cell), Paragraph("Solid", table_cell), Paragraph("White", table_cell), Paragraph("Device is powered on and functioning normally.", table_cell)],
        [Paragraph("POWER", table_cell), Paragraph("Blinking", table_cell), Paragraph("White", table_cell), Paragraph("System is booting up or firmware update is currently in progress. Do not disconnect power.", table_cell)],
        [Paragraph("POWER", table_cell), Paragraph("Solid", table_cell), Paragraph("Amber", table_cell), Paragraph("Hardware self-test failed or system is in recovery boot mode.", table_cell)],
        [Paragraph("INTERNET (WAN)", table_cell), Paragraph("Solid", table_cell), Paragraph("White", table_cell), Paragraph("WAN port connected and internet access is active.", table_cell)],
        [Paragraph("INTERNET (WAN)", table_cell), Paragraph("Solid", table_cell), Paragraph("Amber", table_cell), Paragraph("WAN physical cable is connected, but no internet connection or IP address acquired from ISP.", table_cell)],
        [Paragraph("INTERNET (WAN)", table_cell), Paragraph("Blinking", table_cell), Paragraph("Red", table_cell), Paragraph("WAN cable disconnected or no carrier signal detected from modem. Error code E-101.", table_cell)],
        [Paragraph("INTERNET (WAN)", table_cell), Paragraph("Off", table_cell), Paragraph("None", table_cell), Paragraph("WAN port is disconnected or Ethernet cable is damaged.", table_cell)],
        [Paragraph("2.4 GHz WI-FI", table_cell), Paragraph("Solid", table_cell), Paragraph("White", table_cell), Paragraph("2.4 GHz wireless radio band is enabled and broadcasting.", table_cell)],
        [Paragraph("2.4 GHz WI-FI", table_cell), Paragraph("Blinking", table_cell), Paragraph("White", table_cell), Paragraph("Wireless data is transmitting or receiving on the 2.4 GHz band.", table_cell)],
        [Paragraph("5 GHz WI-FI", table_cell), Paragraph("Solid", table_cell), Paragraph("White", table_cell), Paragraph("5 GHz wireless radio band is active with 160 MHz high-speed mode enabled.", table_cell)],
        [Paragraph("5 GHz WI-FI", table_cell), Paragraph("Blinking", table_cell), Paragraph("White", table_cell), Paragraph("High-speed data transmission in progress on the 5 GHz band.", table_cell)],
        [Paragraph("WPS", table_cell), Paragraph("Blinking", table_cell), Paragraph("White", table_cell), Paragraph("WPS handshake is in progress (active for 120 seconds).", table_cell)],
        [Paragraph("WPS", table_cell), Paragraph("Solid", table_cell), Paragraph("White", table_cell), Paragraph("WPS connection established successfully (illuminates for 5 seconds).", table_cell)],
        [Paragraph("WPS", table_cell), Paragraph("Blinking", table_cell), Paragraph("Amber", table_cell), Paragraph("WPS connection failed or PIN authentication timed out.", table_cell)],
        [Paragraph("USB", table_cell), Paragraph("Solid", table_cell), Paragraph("White", table_cell), Paragraph("Compatible USB 3.0 storage drive detected and mounted successfully.", table_cell)],
        [Paragraph("USB", table_cell), Paragraph("Blinking", table_cell), Paragraph("Amber", table_cell), Paragraph("USB overcurrent detected or filesystem read error. Error code E-501.", table_cell)],
    ]
    t_led = Table(led_table_data, colWidths=[90, 60, 55, 275])
    t_led.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9FA")]),
    ]))
    story.append(t_led)
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 1.3: REAR PANEL PORTS & BUTTONS
    # =========================================================================
    story.append(Paragraph("1.3 Rear Panel Ports and Buttons", h2_style))
    story.append(Paragraph(
        "The rear panel contains all physical data ports, power connectors, and hardware buttons:", body_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph("• <b>WAN Port (1x 10/100/1000 Mbps RJ-45, Color: Blue):</b> Connects to your broadband modem or fiber ONT terminal.", bullet_style))
    story.append(Paragraph("• <b>LAN Ports 1-4 (4x 10/100/1000 Mbps RJ-45, Color: Yellow):</b> Connect wired client devices such as desktop PCs, gaming consoles, NAS appliances, or smart TVs.", bullet_style))
    story.append(Paragraph("• <b>USB 3.0 SuperSpeed Port (Color: Blue):</b> Connect external USB storage drives or printers for local network sharing (supports FAT32, NTFS, exFAT).", bullet_style))
    story.append(Paragraph("• <b>Power Jack (12 V DC, 2.0 A):</b> Connect the bundled AC power adapter.", bullet_style))
    story.append(Paragraph("• <b>Power Button (Latching Push-Switch):</b> Press to toggle power on or off.", bullet_style))
    story.append(Paragraph("• <b>WPS Button:</b> Press and hold for 2 seconds to initiate WPS client pairing.", bullet_style))
    story.append(Paragraph("• <b>Reset Button (Recessed Pin-Hole):</b> Used for factory reset. Press and hold for 10 seconds while the router is powered on until all LEDs flash amber.", bullet_style))
    story.append(Spacer(1, 12))

    # =========================================================================
    # CHAPTER 2: QUICK START & HARDWARE INSTALLATION
    # =========================================================================
    story.append(Paragraph("2. Quick Start and Hardware Installation", h1_style))
    story.append(Paragraph(
        "Follow these sequential steps to properly install and cable your Aurora AX3000 router.", body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("2.1 Connecting the Hardware", h2_style))
    story.append(Paragraph("Step 1: Power off your broadband modem by disconnecting its power adapter. If your modem has a backup battery, remove the battery as well.", step_style))
    story.append(Paragraph("Step 2: Connect one end of the provided RJ-45 Ethernet cable to the Ethernet port on your modem, and connect the other end to the blue <b>WAN port</b> on the Aurora AX3000 router.", step_style))
    story.append(Paragraph("Step 3: Reconnect the power adapter to your modem and wait approximately 2 minutes until all modem status indicators stabilize.", step_style))
    story.append(Paragraph("Step 4: Connect the Aurora 12 V power adapter to the router's DC power port and plug it into an electrical wall outlet. Push the Power button on the rear panel.", step_style))
    story.append(Paragraph("Step 5: Verify that the <b>POWER LED</b> turns solid white, and the <b>INTERNET LED</b> illuminates solid white. This confirms successful WAN link negotiation.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("2.2 Establishing Connection", h2_style))
    story.append(Paragraph("You can connect to the router using either a wired Ethernet cable or wireless Wi-Fi:", body_style))
    story.append(Paragraph("• <b>Wired Connection:</b> Plug an Ethernet cable into any of the yellow <b>LAN ports (1-4)</b> and connect the other end to your computer's Ethernet port.", bullet_style))
    story.append(Paragraph("• <b>Wireless Connection:</b> Search for available wireless networks on your client device. Locate the default SSIDs printed on the product barcode label located on the underside of the router:", bullet_style))
    story.append(Paragraph("     - Default 2.4 GHz SSID: <b>Aurora_AX3000_2.4G</b>", bullet_style))
    story.append(Paragraph("     - Default 5 GHz SSID: <b>Aurora_AX3000_5G</b>", bullet_style))
    story.append(Paragraph("     - Default Wi-Fi Password (PIN): <b>aurora99pass</b> (or 8-digit PIN on the label)", bullet_style))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 3: WEB GUI & INITIAL SETUP WIZARD
    # =========================================================================
    story.append(Paragraph("3. Web GUI and Initial Setup Wizard", h1_style))
    story.append(Paragraph(
        "The router features an intuitive web-based management graphical interface that works in all modern web "
        "browsers including Chrome, Firefox, Safari, and Edge.", body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("3.1 Accessing the Web Management Console", h2_style))
    story.append(Paragraph("Step 1: Open a web browser on a device connected to the router via Wi-Fi or Ethernet.", step_style))
    story.append(Paragraph("Step 2: Enter <b>http://192.168.1.1</b> or <b>http://router.aurora.local</b> in the address bar and press Enter.", step_style))
    story.append(Paragraph("Step 3: On the login screen, enter the default credentials:", step_style))
    story.append(Paragraph("     - Default Username: <b>admin</b>", bullet_style))
    story.append(Paragraph("     - Default Password: <b>admin</b>", bullet_style))
    story.append(Paragraph("Step 4: The system will automatically launch the Quick Setup Wizard upon your first successful login.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.2 Internet Connection Configuration", h2_style))
    story.append(Paragraph("The Quick Setup Wizard guides you through selecting your ISP internet connection type:", body_style))
    story.append(Paragraph("• <b>Dynamic IP (DHCP):</b> Standard for most cable and fiber internet service providers. The router automatically obtains an IP address, subnet mask, and DNS servers from the ISP.", bullet_style))
    story.append(Paragraph("• <b>PPPoE (Point-to-Point Protocol over Ethernet):</b> Common for DSL and certain fiber providers. Requires entering your ISP-assigned Username and Password.", bullet_style))
    story.append(Paragraph("• <b>Static IP:</b> Used when your ISP assigns a fixed, permanent public IP address. Enter the IP Address, Subnet Mask, Default Gateway, Primary DNS (e.g., 8.8.8.8), and Secondary DNS (e.g., 1.1.1.1).", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("3.3 Administrator Account Protection", h2_style))
    story.append(Paragraph("For security compliance, you must change the default administrator password during initial setup.", body_style))
    story.append(Paragraph("Step 1: Navigate to <b>Settings > System Tools > Administration</b>.", step_style))
    story.append(Paragraph("Step 2: Enter the current password (<b>admin</b>).", step_style))
    story.append(Paragraph("Step 3: Enter a strong new administrator password containing at least 8 characters, with a mix of uppercase letters, lowercase letters, numbers, and symbols.", step_style))
    story.append(Paragraph("Step 4: Click <b>Save and Apply</b>. You will be redirected to log in with your new credentials.", step_style))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 4: WIRELESS NETWORK CONFIGURATION
    # =========================================================================
    story.append(Paragraph("4. Wireless Network Configuration", h1_style))
    story.append(Paragraph(
        "Proper wireless configuration ensures optimal throughput, minimal interference, and strong data security.", body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("4.1 Dual-Band Wi-Fi Settings (2.4 GHz and 5 GHz)", h2_style))
    story.append(Paragraph("Navigate to <b>Settings > Wireless > Basic Settings</b> to configure radio parameters:", body_style))
    story.append(Paragraph("• <b>Smart Connect (Band Steering):</b> When enabled, combines 2.4 GHz and 5 GHz under a single unified SSID. The router automatically steers client devices to the best band based on signal strength and congestion.", bullet_style))
    story.append(Paragraph("• <b>2.4 GHz Band Configuration:</b>", bullet_style))
    story.append(Paragraph("     - Channel Width: Auto (20/40 MHz). Select 20 MHz in crowded apartment buildings to prevent overlapping channel collisions.", bullet_style))
    story.append(Paragraph("     - Wireless Channel: Auto (recommended), or manually select channels 1, 6, or 11.", bullet_style))
    story.append(Paragraph("• <b>5 GHz Band Configuration:</b>", bullet_style))
    story.append(Paragraph("     - Channel Width: Up to 160 MHz for maximum Wi-Fi 6 theoretical throughput (2402 Mbps).", bullet_style))
    story.append(Paragraph("     - Wireless Channel: Auto, or manually choose non-DFS channels (36, 40, 44, 48) or DFS channels (52-144).", bullet_style))
    story.append(Paragraph("     - Transmit Power: 100% (High), 50% (Medium), or 25% (Low).", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4.2 Guest Network Setup and Isolation", h2_style))
    story.append(Paragraph(
        "A Guest Network provides visitors with isolated internet access without exposing your primary local network, "
        "NAS storage, printers, or smart IoT devices.", body_style
    ))
    story.append(Paragraph("To enable and configure the Guest Network:", body_style))
    story.append(Paragraph("Step 1: Go to <b>Settings > Wireless > Guest Network</b>.", step_style))
    story.append(Paragraph("Step 2: Toggle the <b>Enable Guest Network</b> switch to ON.", step_style))
    story.append(Paragraph("Step 3: Set a custom Guest SSID (e.g., <b>Aurora_Guest_Wi-Fi</b>).", step_style))
    story.append(Paragraph("Step 4: Select Security Mode: <b>WPA2-PSK (AES)</b> or <b>WPA3-Personal</b>, and enter a dedicated guest passphrase.", step_style))
    story.append(Paragraph("Step 5: Ensure <b>Isolate from Local Network (AP Isolation)</b> is checked. This blocks guests from accessing internal LAN clients at 192.168.1.x.", step_style))
    story.append(Paragraph("Step 6: (Optional) Set Bandwidth Control: Limit maximum download speed (e.g., 20 Mbps) and upload speed (e.g., 5 Mbps).", step_style))
    story.append(Paragraph("Step 7: Click <b>Save</b> to activate the guest network.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4.3 Wi-Fi Security Protocols (WPA2 vs WPA3)", h2_style))
    story.append(Paragraph(
        "The router supports modern encryption protocols under <b>Settings > Wireless > Wireless Security</b>:", body_style
    ))
    story.append(Paragraph("• <b>WPA3-Personal (WPA3-SAE):</b> The latest Wi-Fi standard. Uses Simultaneous Authentication of Equals (SAE) to protect against dictionary and offline password-guessing attacks. Recommended for modern devices.", bullet_style))
    story.append(Paragraph("• <b>WPA2/WPA3 Mixed Mode:</b> Allows modern WPA3 devices to connect securely while maintaining backwards compatibility for older legacy devices.", bullet_style))
    story.append(Paragraph("• <b>WPA2-PSK (AES):</b> Universal standard for older smartphones, printers, and IoT devices.", bullet_style))
    story.append(Paragraph("• <b>Open / None:</b> Unencrypted. Strongly discouraged for home environments.", warning_style))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 5: ADVANCED NETWORK MANAGEMENT
    # =========================================================================
    story.append(Paragraph("5. Advanced Network Management", h1_style))
    story.append(Paragraph(
        "The Aurora AX3000 includes enterprise-grade management features including QoS, Parental Controls, "
        "NAT forwarding, and built-in VPN server capabilities.", body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("5.1 Quality of Service (QoS) Bandwidth Prioritization", h2_style))
    story.append(Paragraph(
        "Quality of Service (QoS) guarantees critical network traffic receives top priority during high-congestion periods. "
        "Navigate to <b>Settings > Advanced > QoS</b>.", body_style
    ))
    story.append(Paragraph("Step 1: Toggle <b>Enable QoS</b> to ON.", step_style))
    story.append(Paragraph("Step 2: Enter your actual broadband ISP bandwidth (e.g., 500 Mbps Download, 50 Mbps Upload). Running the built-in Speed Test tool can automatically detect these values.", step_style))
    story.append(Paragraph("Step 3: Select a prioritization preset:", step_style))
    story.append(Paragraph("     - <b>Gaming Priority:</b> Minimizes packet latency and bufferbloat for online gaming consoles and PCs.", bullet_style))
    story.append(Paragraph("     - <b>Streaming Priority:</b> Prioritizes 4K/8K video streaming protocols (HLS, DASH).", bullet_style))
    story.append(Paragraph("     - <b>Work From Home:</b> Prioritizes Zoom, Microsoft Teams, VoIP, and VPN traffic.", bullet_style))
    story.append(Paragraph("Step 4: You can also drag connected devices into the High Priority category to guarantee dedicated bandwidth.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5.2 Parental Controls and Domain Filtering", h2_style))
    story.append(Paragraph(
        "Protect family members and manage screen time via <b>Settings > Parental Controls</b>:", body_style
    ))
    story.append(Paragraph("Step 1: Click <b>Add Profile</b>, enter a profile name (e.g., 'Kids Devices'), and select the target client MAC addresses.", step_style))
    story.append(Paragraph("Step 2: Configure <b>Time Limits & Bedtime Schedule</b>: Specify allowed internet access hours per day (e.g., Mon-Fri 4:00 PM to 8:30 PM, Sat-Sun 8:00 AM to 9:30 PM). Internet access is suspended outside these intervals.", step_style))
    story.append(Paragraph("Step 3: Configure <b>Content Filter</b>: Select content categories to block (Adult Content, Gambling, Social Media, Online Gaming).", step_style))
    story.append(Paragraph("Step 4: Configure <b>Domain Blacklist / Whitelist</b>: Add specific website URLs to permanently block or allow.", step_style))
    story.append(Paragraph("Step 5: Click <b>Save</b> to enforce the rules immediately.", step_style))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 5.3 - 5.5: NAT, VPN, USB
    # =========================================================================
    story.append(Paragraph("5.3 Port Forwarding, NAT and DMZ Rules", h2_style))
    story.append(Paragraph(
        "Port forwarding permits external internet traffic on specified ports to reach specific servers on your local network.", body_style
    ))
    story.append(Paragraph("To create a Port Forwarding rule:", body_style))
    story.append(Paragraph("Step 1: Navigate to <b>Settings > Advanced > NAT Forwarding > Virtual Servers</b>.", step_style))
    story.append(Paragraph("Step 2: Click <b>Add Rule</b>.", step_style))
    story.append(Paragraph("Step 3: Enter the <b>Service Name</b> (e.g., 'Web Server' or 'Minecraft').", step_style))
    story.append(Paragraph("Step 4: Enter the <b>External Port</b> (e.g., 8080) and <b>Internal Port</b> (e.g., 80).", step_style))
    story.append(Paragraph("Step 5: Enter the <b>Internal IP Address</b> of the target host (e.g., 192.168.1.150). We recommend assigning a static DHCP reservation under <b>Settings > Network > DHCP Server</b> first.", step_style))
    story.append(Paragraph("Step 6: Select the <b>Protocol</b>: TCP, UDP, or ALL.", step_style))
    story.append(Paragraph("Step 7: Check <b>Enable this entry</b> and click <b>Save</b>.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5.4 VPN Server (OpenVPN and WireGuard)", h2_style))
    story.append(Paragraph(
        "The router includes integrated OpenVPN and WireGuard servers, allowing you to securely connect to your home "
        "network from anywhere over an encrypted tunnel. Navigate to <b>Settings > Advanced > VPN Server</b>.", body_style
    ))
    story.append(Paragraph("Step 1: Select <b>OpenVPN</b> or <b>WireGuard</b> and toggle <b>Enable VPN Server</b> to ON.", step_style))
    story.append(Paragraph("Step 2: Select Service Port (default: 1194 for OpenVPN UDP, 51820 for WireGuard).", step_style))
    story.append(Paragraph("Step 3: Select Client Access: <b>Home Network and Internet</b> or <b>Home Network Only</b>.", step_style))
    story.append(Paragraph("Step 4: Click <b>Generate Certificate / Configuration</b> and download the client profile (`client.ovpn` or `wg0.conf`).", step_style))
    story.append(Paragraph("Step 5: Import the profile into the OpenVPN or WireGuard client on your laptop or smartphone.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("5.5 USB 3.0 Storage and Media Server", h2_style))
    story.append(Paragraph(
        "Connect an external USB hard drive or flash drive to the rear USB 3.0 port. Navigate to <b>Settings > USB Sharing</b>:", body_style
    ))
    story.append(Paragraph("• <b>Network Neighborhood (Samba / SMB):</b> Access files from Windows Explorer at <b>\\\\192.168.1.1\\share</b> or macOS Finder via <b>smb://192.168.1.1/share</b>.", bullet_style))
    story.append(Paragraph("• <b>Media Server (DLNA):</b> Stream video, photos, and music directly to smart TVs, game consoles, and media boxes.", bullet_style))
    story.append(Paragraph("• <b>FTP Server:</b> Enables encrypted local and remote FTP access (port 21).", bullet_style))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 6: SYSTEM MAINTENANCE & RECOVERY
    # =========================================================================
    story.append(Paragraph("6. System Maintenance and Recovery", h1_style))
    story.append(Paragraph(
        "Regular maintenance ensures peak performance, security patch compliance, and configuration preservation.", body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("6.1 Firmware Upgrade Procedures", h2_style))
    story.append(Paragraph(
        "Firmware upgrades provide feature enhancements, bug fixes, and security patches. Do not power off the router "
        "or disconnect network cables during a firmware upgrade.", warning_style
    ))
    story.append(Paragraph("<b>Method 1: Online Automatic Firmware Upgrade</b>", body_style))
    story.append(Paragraph("Step 1: Navigate to <b>Settings > System Tools > Firmware Update</b>.", step_style))
    story.append(Paragraph("Step 2: Click <b>Check for Updates</b>. The router will contact the Aurora cloud server to determine if a newer firmware build is available.", step_style))
    story.append(Paragraph("Step 3: If an update is detected, click <b>Download and Upgrade</b>. The router will download the package, verify the digital signature, and flash the memory.", step_style))
    story.append(Paragraph("Step 4: The system will reboot automatically within 3 minutes. The POWER LED will blink white during writing and return to solid white when complete.", step_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Method 2: Manual Local Firmware File Upgrade</b>", body_style))
    story.append(Paragraph("Step 1: Download the official firmware image file (e.g., `AX3000_v1.4.2_build202610.bin`) from `https://aurora-tech.com/support`.", step_style))
    story.append(Paragraph("Step 2: In the web interface under <b>Settings > System Tools > Firmware Update</b>, click <b>Browse</b> and select the `.bin` file.", step_style))
    story.append(Paragraph("Step 3: Click <b>Upgrade</b> and confirm the prompt.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6.2 Backup and Restoration of Configuration", h2_style))
    story.append(Paragraph(
        "Before modifying advanced settings or upgrading firmware, back up your current configuration file:", body_style
    ))
    story.append(Paragraph("• <b>Backing Up:</b> Go to <b>Settings > System Tools > Backup & Restore</b> and click <b>Backup</b>. The router downloads an encrypted `.cfg` file to your computer.", bullet_style))
    story.append(Paragraph("• <b>Restoring:</b> Under the same menu, click <b>Browse</b>, select your saved `.cfg` file, and click <b>Restore</b>. The router will reload settings and reboot.", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("6.3 Factory Reset and Reboot Methods", h2_style))
    story.append(Paragraph(
        "A factory reset restores all settings to default values, erasing custom passwords, SSIDs, and rules.", body_style
    ))
    story.append(Paragraph("• <b>Soft Reset via Web GUI:</b> Navigate to <b>Settings > System Tools > Backup & Restore</b> and click <b>Factory Restore</b>.", bullet_style))
    story.append(Paragraph("• <b>Hard Reset via Physical Button:</b> Locate the recessed <b>Reset</b> button on the rear panel. While the router is powered on, use a straightened paperclip to press and hold the button for <b>10 seconds</b> until all front panel LEDs illuminate amber simultaneously and then turn off. Release the button. The router will reboot in default state.", bullet_style))
    story.append(Paragraph("• <b>Standard System Reboot:</b> Go to <b>Settings > System Tools > Reboot</b> and click <b>Reboot</b>, or momentarily press the rear power switch off and on.", bullet_style))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 7: TROUBLESHOOTING & ERROR CODE DIAGNOSTIC TABLE
    # =========================================================================
    story.append(Paragraph("7. Troubleshooting and Diagnostic Error Codes", h1_style))
    story.append(Paragraph(
        "When an operational failure occurs, the router logs an alphanumeric error code in the system log and displays "
        "it in the Web GUI under <b>Settings > System Tools > System Log</b>. Use the table below for immediate diagnostic resolution.", body_style
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph("7.1 Comprehensive Diagnostic Error Code Table", h2_style))

    error_table_data = [
        [Paragraph("Code", table_header), Paragraph("Error Name", table_header), Paragraph("Root Cause", table_header), Paragraph("Recommended Resolution Steps", table_header)],
        [
            Paragraph("<b>E-101</b>", table_cell),
            Paragraph("WAN Link Disconnected", table_cell),
            Paragraph("No physical carrier signal or Ethernet link on the blue WAN port.", table_cell),
            Paragraph("1. Check that the Ethernet cable is securely clicked into the blue WAN port and the modem.<br/>2. Verify modem is powered on.<br/>3. Test with a different Cat 5e/Cat 6 cable.<br/>4. Power cycle modem and router.", table_cell)
        ],
        [
            Paragraph("<b>E-102</b>", table_cell),
            Paragraph("PPPoE Auth Failure", table_cell),
            Paragraph("The broadband ISP rejected the configured PPPoE credentials.", table_cell),
            Paragraph("1. Navigate to Settings > Network > WAN.<br/>2. Re-enter your ISP username and password.<br/>3. Verify credentials with your ISP.<br/>4. Ensure ISP account has no billing suspension.", table_cell)
        ],
        [
            Paragraph("<b>E-103</b>", table_cell),
            Paragraph("DHCP Lease Timeout", table_cell),
            Paragraph("Router requested an IP lease from ISP gateway but received no response.", table_cell),
            Paragraph("1. Disconnect router WAN cable.<br/>2. Power cycle ISP modem and wait 5 minutes.<br/>3. Reconnect WAN cable.<br/>4. Under Settings > Network > WAN, click 'Release' then 'Renew'.<br/>5. Contact ISP if issue persists.", table_cell)
        ],
        [
            Paragraph("<b>E-201</b>", table_cell),
            Paragraph("IP Address Conflict", table_cell),
            Paragraph("Router LAN IP matches modem or upstream gateway subnet (e.g., both 192.168.1.1).", table_cell),
            Paragraph("1. Navigate to Settings > Network > LAN.<br/>2. Change Router LAN IP to 192.168.2.1 or 10.0.0.1.<br/>3. Set Subnet Mask to 255.255.255.0.<br/>4. Click Save and reboot router.", table_cell)
        ],
        [
            Paragraph("<b>E-204</b>", table_cell),
            Paragraph("DNS Resolution Failure", table_cell),
            Paragraph("Router WAN is connected but cannot resolve domain names into IP addresses.", table_cell),
            Paragraph("1. Go to Settings > Network > WAN.<br/>2. Enable 'Use Custom DNS Servers'.<br/>3. Set Primary DNS to 8.8.8.8 (Google DNS) or 1.1.1.1 (Cloudflare DNS).<br/>4. Set Secondary DNS to 8.8.4.4 or 1.0.0.1.<br/>5. Click Save and Apply.", table_cell)
        ],
        [
            Paragraph("<b>E-301</b>", table_cell),
            Paragraph("Wi-Fi Radio Thermal Fault", table_cell),
            Paragraph("Internal wireless chip temperature exceeded safe threshold (85°C).", table_cell),
            Paragraph("1. Power off router immediately.<br/>2. Relocate router to an open, well-ventilated location away from heat sources.<br/>3. Clear dust from side vents.<br/>4. Allow 15 minutes to cool, then power on.", table_cell)
        ],
        [
            Paragraph("<b>E-305</b>", table_cell),
            Paragraph("Guest Isolation Fault", table_cell),
            Paragraph("Internal VLAN firewall rule failed to isolate guest client traffic.", table_cell),
            Paragraph("1. Go to Settings > Wireless > Guest Network.<br/>2. Toggle Guest Network OFF, click Save.<br/>3. Toggle Guest Network back ON and re-check 'Isolate from Local Network'.<br/>4. Reboot router.", table_cell)
        ],
        [
            Paragraph("<b>E-401</b>", table_cell),
            Paragraph("Firmware Checksum Fault", table_cell),
            Paragraph("Uploaded firmware `.bin` file failed SHA-256 verification or is corrupt.", table_cell),
            Paragraph("1. Delete the downloaded firmware file from your computer.<br/>2. Re-download official file from https://aurora-tech.com/support.<br/>3. Verify file size matches release notes.<br/>4. Retry upgrade over a wired LAN connection.", table_cell)
        ],
        [
            Paragraph("<b>E-402</b>", table_cell),
            Paragraph("Flash Memory Write Error", table_cell),
            Paragraph("Low-level write failure to internal NAND flash memory block.", table_cell),
            Paragraph("1. Do not power off router.<br/>2. Perform Emergency TFTP Firmware Recovery procedure.<br/>3. If recovery fails, submit RMA warranty claim.", table_cell)
        ],
        [
            Paragraph("<b>E-501</b>", table_cell),
            Paragraph("USB Overcurrent Fault", table_cell),
            Paragraph("Connected USB storage or device drew more than 1.5 A power limit.", table_cell),
            Paragraph("1. Unplug the USB device from the rear USB 3.0 port immediately.<br/>2. If using an external 3.5\" hard drive, connect its external AC power supply.<br/>3. Power cycle router.", table_cell)
        ],
    ]
    t_err = Table(error_table_data, colWidths=[50, 95, 125, 210])
    t_err.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9FA")]),
    ]))
    story.append(t_err)
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 7.2: STEP-BY-STEP DIAGNOSTIC PROCEDURES
    # =========================================================================
    story.append(Paragraph("7.2 Diagnostic Procedures for Common Issues", h2_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Problem: No Internet Access (Internet LED is Solid Amber or Blinking Red)</b>", body_style))
    story.append(Paragraph("Follow these sequential troubleshooting steps:", body_style))
    story.append(Paragraph("Step 1: Check the physical WAN Ethernet cable connecting your modem and the blue router WAN port. Ensure both ends click firmly into place.", step_style))
    story.append(Paragraph("Step 2: Power cycle the network: Power off both the modem and the Aurora router. Wait 30 seconds. Power on the modem first and wait until its Online/Internet LED is solid. Then power on the router.", step_style))
    story.append(Paragraph("Step 3: Log in to the Web GUI at `http://192.168.1.1` and check the WAN status on the Network Map. If IP Address is `0.0.0.0`, click 'Renew'.", step_style))
    story.append(Paragraph("Step 4: If your ISP requires MAC binding (common with cable internet), go to <b>Settings > Network > MAC Clone</b>, select <b>Clone Current Computer MAC</b>, and click <b>Save</b>.", step_style))
    story.append(Paragraph("Step 5: If the status displays Error Code E-204, configure static public DNS servers (8.8.8.8 and 1.1.1.1) under WAN settings.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Problem: Wi-Fi Speed is Slower than Expected</b>", body_style))
    story.append(Paragraph("Follow these optimization steps:", body_style))
    story.append(Paragraph("Step 1: Verify your device is connected to the <b>5 GHz band</b> rather than the 2.4 GHz band. The 5 GHz band supports up to 2402 Mbps, whereas 2.4 GHz maxes out at 574 Mbps.", step_style))
    story.append(Paragraph("Step 2: In the Web GUI under <b>Settings > Wireless > 5 GHz</b>, ensure <b>Channel Width</b> is set to <b>160 MHz</b> and <b>Mode</b> is set to <b>802.11ax Only</b> or <b>802.11a/n/ac/ax Mixed</b>.", step_style))
    story.append(Paragraph("Step 3: Relocate the router to an elevated, open area away from concrete walls, metal cabinets, microwave ovens, and 2.4 GHz cordless phones.", step_style))
    story.append(Paragraph("Step 4: Adjust the four external antennas: Orient two antennas vertically at 90 degrees and two at 45 degrees for optimal 3D spatial coverage.", step_style))
    story.append(Paragraph("Step 5: Check <b>Settings > Advanced > QoS</b>. If QoS is enabled with incorrect broadband bandwidth caps, disable QoS or re-run the speed calibration.", step_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Problem: Forgotten Administrator Password</b>", body_style))
    story.append(Paragraph("If you forget your custom Web GUI admin password, you must reset the router to factory defaults:", body_style))
    story.append(Paragraph("Step 1: Keep the router powered on.", step_style))
    story.append(Paragraph("Step 2: Press and hold the recessed <b>Reset</b> button on the rear panel with a pin for 10 seconds until all front panel LEDs flash amber.", step_style))
    story.append(Paragraph("Step 3: Release the button and wait 2 minutes for the router to complete rebooting.", step_style))
    story.append(Paragraph("Step 4: Connect to the default SSID (`Aurora_AX3000_2.4G`) and browse to `http://192.168.1.1`. Log in using default username `admin` and password `admin`.", step_style))
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 8: TECHNICAL SPECIFICATIONS
    # =========================================================================
    story.append(Paragraph("8. Technical Specifications and Hardware Architecture", h1_style))
    story.append(Paragraph(
        "Detailed electrical, physical, software, and radio specifications for the Aurora AX3000-PRO router:", body_style
    ))
    story.append(Spacer(1, 6))

    specs_data = [
        [Paragraph("Category", table_header), Paragraph("Specification Parameter", table_header), Paragraph("Technical Value / Description", table_header)],
        [Paragraph("Processor", table_cell), Paragraph("Main CPU", table_cell), Paragraph("Dual-Core 1.5 GHz ARM Cortex-A53 64-bit Network Processor", table_cell)],
        [Paragraph("Memory", table_cell), Paragraph("System RAM", table_cell), Paragraph("512 MB DDR3 High-Speed Memory", table_cell)],
        [Paragraph("Memory", table_cell), Paragraph("Flash Storage", table_cell), Paragraph("256 MB SPI NAND Flash", table_cell)],
        [Paragraph("Wireless", table_cell), Paragraph("Standards", table_cell), Paragraph("IEEE 802.11ax/ac/n/a 5 GHz, IEEE 802.11ax/n/b/g 2.4 GHz", table_cell)],
        [Paragraph("Wireless", table_cell), Paragraph("Signal Rate", table_cell), Paragraph("5 GHz: up to 2402 Mbps (160 MHz); 2.4 GHz: up to 574 Mbps", table_cell)],
        [Paragraph("Wireless", table_cell), Paragraph("Modulation", table_cell), Paragraph("1024-QAM, OFDMA, DL/UL MU-MIMO, BSS Coloring, Beamforming", table_cell)],
        [Paragraph("Wireless", table_cell), Paragraph("Security", table_cell), Paragraph("WPA3-Personal (SAE), WPA2-PSK (AES), WPA-Enterprise, WPS 2.0", table_cell)],
        [Paragraph("Antennas", table_cell), Paragraph("Antenna Array", table_cell), Paragraph("4x External High-Gain Omnidirectional Antennas (5 dBi gain each)", table_cell)],
        [Paragraph("Ethernet Ports", table_cell), Paragraph("WAN Interface", table_cell), Paragraph("1x 10/100/1000 Mbps Gigabit RJ-45 (Auto MDI/MDIX)", table_cell)],
        [Paragraph("Ethernet Ports", table_cell), Paragraph("LAN Interface", table_cell), Paragraph("4x 10/100/1000 Mbps Gigabit RJ-45 (Auto MDI/MDIX)", table_cell)],
        [Paragraph("USB Expansion", table_cell), Paragraph("USB Interface", table_cell), Paragraph("1x USB 3.0 Type-A SuperSpeed (5 Gbps, 5V / 1.5A maximum output)", table_cell)],
        [Paragraph("Buttons", table_cell), Paragraph("Hardware Controls", table_cell), Paragraph("Power On/Off, Reset (recessed pin-hole), WPS button", table_cell)],
        [Paragraph("Power", table_cell), Paragraph("Power Supply", table_cell), Paragraph("External AC/DC Adapter; Input: 100-240V 50/60Hz; Output: 12 V DC, 2.0 A", table_cell)],
        [Paragraph("Power", table_cell), Paragraph("Power Consumption", table_cell), Paragraph("Max 18.5 W (with active USB 3.0 load); Typical: 8.2 W idle", table_cell)],
        [Paragraph("Physical", table_cell), Paragraph("Dimensions (W x D x H)", table_cell), Paragraph("260.2 x 135.0 x 38.6 mm (10.2 x 5.3 x 1.5 in) without antennas", table_cell)],
        [Paragraph("Physical", table_cell), Paragraph("Weight", table_cell), Paragraph("485 g (1.07 lbs)", table_cell)],
        [Paragraph("Environment", table_cell), Paragraph("Operating Temperature", table_cell), Paragraph("0°C to 40°C (32°F to 104°F)", table_cell)],
        [Paragraph("Environment", table_cell), Paragraph("Operating Humidity", table_cell), Paragraph("10% to 90% Non-Condensing", table_cell)],
    ]
    t_specs = Table(specs_data, colWidths=[80, 140, 260])
    t_specs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B2545")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8F9FA")]),
    ]))
    story.append(t_specs)
    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 9: SAFETY, WARRANTY & SUPPORT CONTACTS
    # =========================================================================
    story.append(Paragraph("9. Regulatory Compliance, Safety and Warranty", h1_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("9.1 Safety and Handling Precautions", h2_style))
    story.append(Paragraph("• Do not expose this device to water, moisture, rain, liquids, or high-humidity environments.", bullet_style))
    story.append(Paragraph("• Do not place the router near heat sources such as radiators, heat registers, stoves, or amplifiers.", bullet_style))
    story.append(Paragraph("• Maintain at least 10 cm (4 inches) of clear space around the ventilation openings to allow sufficient airflow.", bullet_style))
    story.append(Paragraph("• Use only the supplied 12 V, 2.0 A power adapter. Using an uncertified power supply voids warranty and risks fire or electric shock.", bullet_style))
    story.append(Paragraph("• Do not attempt to open the casing or service internal electronics yourself. Contact authorized technicians.", bullet_style))
    story.append(Spacer(1, 8))

    story.append(Paragraph("9.2 Regulatory Statements", h2_style))
    story.append(Paragraph(
        "<b>FCC Statement (Part 15 Class B):</b> This equipment has been tested and found to comply with the limits for a "
        "Class B digital device, pursuant to Part 15 of the FCC Rules. Operation is subject to the following two conditions: "
        "(1) This device may not cause harmful interference, and (2) this device must accept any interference received, "
        "including interference that may cause undesired operation.", body_style
    ))
    story.append(Paragraph(
        "<b>CE Mark Warning:</b> This is a Class B product. In a domestic environment, this product may cause radio interference, "
        "in which case the user may be required to take adequate measures. Operations in the 5.15-5.35 GHz band are restricted "
        "to indoor usage in all EU member states.", body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("9.3 Two-Year Limited Hardware Warranty", h2_style))
    story.append(Paragraph(
        "Aurora Networks Inc. warrants this Aurora AX3000 Wi-Fi Router against defects in materials and workmanship under "
        "normal consumer use for a period of <b>two (2) years</b> from the original retail purchase date. "
        "During this warranty period, Aurora Networks will, at its option, either repair the product using new or refurbished parts, "
        "or replace the product with a new or reconditioned unit of equal or greater functionality.", body_style
    ))
    story.append(Paragraph(
        "<b>Warranty Exclusions:</b> This warranty does not cover: (a) cosmetic damage including scratches and dents; "
        "(b) damage caused by accident, abuse, liquid contact, fire, power surges, or lightning strikes; "
        "(c) damage caused by operating outside environmental specifications; (d) unauthorized third-party firmware modification "
        "(e.g., flashing custom OpenWrt or DD-WRT builds).", body_style
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("9.4 Customer Support Contacts", h2_style))
    story.append(Paragraph("For technical assistance, driver downloads, warranty claims, and knowledgebase articles:", body_style))
    story.append(Paragraph("• <b>Technical Support Phone:</b> +1-800-555-0199 (Toll-Free USA & Canada)", bullet_style))
    story.append(Paragraph("• <b>International Support:</b> +1-415-555-0142", bullet_style))
    story.append(Paragraph("• <b>Support Email:</b> support@aurora-tech.com", bullet_style))
    story.append(Paragraph("• <b>Warranty Claims Portal:</b> https://aurora-tech.com/warranty", bullet_style))
    story.append(Paragraph("• <b>Firmware & Documentation Downloads:</b> https://aurora-tech.com/support/ax3000", bullet_style))
    story.append(Paragraph("• <b>Support Hours:</b> Monday through Friday, 8:00 AM to 8:00 PM Eastern Standard Time (EST)", bullet_style))
    story.append(Paragraph("• <b>Corporate Headquarters:</b> Aurora Networks Inc., 450 Innovation Parkway, Suite 300, San Jose, CA 95134, USA", bullet_style))

    # Build the document using our custom NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Manual PDF successfully generated at: {output_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_file = out_dir / "aurora_ax3000_manual.pdf"
    build_manual_pdf(str(out_file))

