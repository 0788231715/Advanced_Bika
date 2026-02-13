**The Bika System: AI-Enhanced Inventory & Quality Management**

The system is a comprehensive, AI-driven platform designed to manage and optimize produce inventory, focusing heavily on quality control, shelf-life management, and predictive insights. It supports users and product lifecycle from registration through proactive operational interventions.

**Key Capabilities:**

1.  **Comprehensive Product & Batch Management:**
    *   Tracks `Product` and `FruitBatch` entities, including detailed information about each fruit/produce type and specific batches.
    *   Manages `StorageLocation`s where products are stored.

2.  **Robust User Management:**
    *   Utilizes a `CustomUser` model, implying support for various roles (e.g., administrators, inventory managers, quality control personnel).
    *   Incorporates role-based notifications, specifically emailing `admin_users` for critical issues.

3.  **Dynamic AI Model Management:**
    *   **Flexible Model Loading:** The `EnhancedBikaAIService` dynamically loads and manages multiple active AI models from `TrainedModel` records in the database.
    *   **Extensible Design:** Supports various model types, including `FruitQualityPredictor`, `FruitRipenessPredictor`, `FruitDiseasePredictor`, and `FruitPricePredictor`, allowing for easy expansion with new AI capabilities.

4.  **Proactive Monitoring & Predictive Analytics (Core AI Functionality):**
    *   **Real-time Sensor Data Integration:** Collects and processes real-time sensor data (temperature, humidity, light intensity, CO2 levels, ethylene levels) from `StorageLocation`s or directly from product attributes, indicating IoT integration.
    *   **Multi-faceted AI Predictions:** Generates critical insights across several domains:
        *   **Product Quality:** Predicts the quality grade (e.g., Rotten, Poor, Fair).
        *   **Ripeness:** Predicts the ripeness stage of fruits.
        *   **Disease Risk:** Assesses the likelihood of disease outbreaks.
        *   **Pricing:** Provides data-driven price recommendations.
        *   **Ethylene Monitoring:** (Implied) Detects and monitors ethylene levels, crucial for ripening control.
    *   **Historical Prediction Tracking:** All predictions are saved in the `FruitPrediction` model, capturing predicted values, confidence scores, input sensor conditions, and associated metadata for later analysis and auditing.

5.  **Automated Alerting & Notification System:**
    *   **Intelligent Alert Generation:** Automatically creates `AlertNotification`s when prediction results indicate potential problems (e.g., deteriorating quality, high disease risk, environmental anomalies).
    *   **Prioritized Alerts:** Alerts are categorized by priority (critical, high, medium, info) to guide immediate attention.
    *   **Actionable Recommendations:** Alerts come with specific messages and recommendations (e.g., "Immediate disposal," "Offer discount," "Quarantine affected products") to facilitate prompt decision-making.
    *   **Persistent Alert Records:** All alerts are saved to the `AlertNotification` model, linked to the relevant product, batch, and prediction for comprehensive record-keeping.
    *   **Email-based Critical Notifications:** Critical and high-priority alerts trigger automated email notifications to designated `admin_users`, ensuring timely awareness and intervention.

6.  **Robust Data Persistence & Integrity:**
    *   Utilizes Django's Object-Relational Mapper (ORM) for secure and efficient storage of all operational and AI-generated data.
    *   Employs atomic transactions (`transaction.atomic()`) to ensure data consistency during critical write operations.

7.  **Operational Logging:**
    *   Maintains detailed logs (`logging` module) for all significant events, including model loading, prediction processes, and alert generation, vital for system monitoring, debugging, and performance analysis.

**User and Product Lifecycle Workflow:**

*   **Initialization & Setup:**
    *   **User Registration:** Staff or management users are registered, defining their roles and permissions.
    *   **Product/Batch/Location Definition:** Products are cataloged, batches are created, and storage locations are established.
    *   **Sensor Deployment:** (Physical infrastructure) Sensors are integrated to feed data into the system.
    *   **AI Model Training & Configuration:** AI models are trained (potentially off-system or via management commands) and then configured within the system via `TrainedModel` records to become active.

*   **Continuous Operation:**
    *   **Data Ingestion:** Sensor data is continuously fed into the system.
    *   **AI-Powered Analysis:** The `EnhancedBikaAIService` constantly processes this data, runs predictions across all active AI models, and generates insights.
    *   **Proactive Alerting:** Based on these insights, the system identifies anomalies or risks and automatically generates prioritized alerts.
    *   **Notification & Action:** `admin_users` receive notifications for critical issues, access the system to review detailed predictions and recommendations, and initiate corrective actions (e.g., moving products, repricing, adjusting environmental controls).

*   **Completion of Ability & Improvement:**
    *   **Action Execution:** Users implement the recommended actions.
    *   **Performance Monitoring:** The system continuously tracks the performance of its AI models (via `get_model_performance`) and the effectiveness of interventions.
    *   **Auditing & Reporting:** Historical data on predictions and alerts provides a rich audit trail and enables comprehensive reporting on product quality, operational efficiency, and waste reduction over time, feeding back into further system and model improvements.

In essence, the Bika system acts as an intelligent assistant for produce management, automating the detection of potential issues and providing actionable intelligence to optimize product handling, minimize losses, and ensure quality throughout the supply chain.