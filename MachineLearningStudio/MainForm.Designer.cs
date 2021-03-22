
namespace MachineLearningStudio
{
   partial class MainForm
   {
      /// <summary>
      /// Variabile di progettazione necessaria.
      /// </summary>
      private System.ComponentModel.IContainer components = null;

      /// <summary>
      /// Pulire le risorse in uso.
      /// </summary>
      /// <param name="disposing">ha valore true se le risorse gestite devono essere eliminate, false in caso contrario.</param>
      protected override void Dispose(bool disposing)
      {
         if (disposing && (components != null)) {
            components.Dispose();
         }
         base.Dispose(disposing);
      }

      #region Codice generato da Progettazione Windows Form

      /// <summary>
      /// Metodo necessario per il supporto della finestra di progettazione. Non modificare
      /// il contenuto del metodo con l'editor di codice.
      /// </summary>
      private void InitializeComponent()
      {
         this.tabControlMain = new System.Windows.Forms.TabControl();
         this.tabPageTextMeaning = new System.Windows.Forms.TabPage();
         this.pageTextMeaning = new MachineLearningStudio.PageTextMeaning();
         this.tabPageObjectDetection = new System.Windows.Forms.TabPage();
         this.tabPageImageClassification = new System.Windows.Forms.TabPage();
         this.pageImageClassification = new MachineLearningStudio.PageImageClassification();
         this.tabPageFeetRegression = new System.Windows.Forms.TabPage();
         this.pageFeetRegression = new MachineLearningStudio.PageFeetRegression();
         this.pageObjectDetection = new MachineLearningStudio.PageObjectDetection();
         this.tabControlMain.SuspendLayout();
         this.tabPageTextMeaning.SuspendLayout();
         this.tabPageObjectDetection.SuspendLayout();
         this.tabPageImageClassification.SuspendLayout();
         this.tabPageFeetRegression.SuspendLayout();
         this.SuspendLayout();
         // 
         // tabControlMain
         // 
         this.tabControlMain.Controls.Add(this.tabPageTextMeaning);
         this.tabControlMain.Controls.Add(this.tabPageObjectDetection);
         this.tabControlMain.Controls.Add(this.tabPageImageClassification);
         this.tabControlMain.Controls.Add(this.tabPageFeetRegression);
         this.tabControlMain.Dock = System.Windows.Forms.DockStyle.Fill;
         this.tabControlMain.Location = new System.Drawing.Point(0, 0);
         this.tabControlMain.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.tabControlMain.Name = "tabControlMain";
         this.tabControlMain.SelectedIndex = 0;
         this.tabControlMain.Size = new System.Drawing.Size(1165, 770);
         this.tabControlMain.TabIndex = 5;
         // 
         // tabPageTextMeaning
         // 
         this.tabPageTextMeaning.Controls.Add(this.pageTextMeaning);
         this.tabPageTextMeaning.Location = new System.Drawing.Point(4, 24);
         this.tabPageTextMeaning.Name = "tabPageTextMeaning";
         this.tabPageTextMeaning.Size = new System.Drawing.Size(1157, 742);
         this.tabPageTextMeaning.TabIndex = 5;
         this.tabPageTextMeaning.Text = "Text meaning";
         this.tabPageTextMeaning.UseVisualStyleBackColor = true;
         // 
         // pageTextMeaning
         // 
         this.pageTextMeaning.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pageTextMeaning.Location = new System.Drawing.Point(0, 0);
         this.pageTextMeaning.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.pageTextMeaning.Name = "pageTextMeaning";
         this.pageTextMeaning.Size = new System.Drawing.Size(1157, 742);
         this.pageTextMeaning.TabIndex = 0;
         // 
         // tabPageObjectDetection
         // 
         this.tabPageObjectDetection.Controls.Add(this.pageObjectDetection);
         this.tabPageObjectDetection.Location = new System.Drawing.Point(4, 24);
         this.tabPageObjectDetection.Name = "tabPageObjectDetection";
         this.tabPageObjectDetection.Size = new System.Drawing.Size(1157, 742);
         this.tabPageObjectDetection.TabIndex = 6;
         this.tabPageObjectDetection.Text = "Object detection";
         this.tabPageObjectDetection.UseVisualStyleBackColor = true;
         // 
         // tabPageImageClassification
         // 
         this.tabPageImageClassification.Controls.Add(this.pageImageClassification);
         this.tabPageImageClassification.Location = new System.Drawing.Point(4, 24);
         this.tabPageImageClassification.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.tabPageImageClassification.Name = "tabPageImageClassification";
         this.tabPageImageClassification.Size = new System.Drawing.Size(1157, 742);
         this.tabPageImageClassification.TabIndex = 3;
         this.tabPageImageClassification.Text = "Image classification";
         this.tabPageImageClassification.UseVisualStyleBackColor = true;
         // 
         // pageImageClassification
         // 
         this.pageImageClassification.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pageImageClassification.Location = new System.Drawing.Point(0, 0);
         this.pageImageClassification.Margin = new System.Windows.Forms.Padding(5, 3, 5, 3);
         this.pageImageClassification.Name = "pageImageClassification";
         this.pageImageClassification.Size = new System.Drawing.Size(1157, 742);
         this.pageImageClassification.TabIndex = 0;
         // 
         // tabPageFeetRegression
         // 
         this.tabPageFeetRegression.Controls.Add(this.pageFeetRegression);
         this.tabPageFeetRegression.Location = new System.Drawing.Point(4, 24);
         this.tabPageFeetRegression.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.tabPageFeetRegression.Name = "tabPageFeetRegression";
         this.tabPageFeetRegression.Size = new System.Drawing.Size(1157, 742);
         this.tabPageFeetRegression.TabIndex = 2;
         this.tabPageFeetRegression.Text = "Feet Regression";
         this.tabPageFeetRegression.UseVisualStyleBackColor = true;
         // 
         // pageFeetRegression
         // 
         this.pageFeetRegression.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pageFeetRegression.Location = new System.Drawing.Point(0, 0);
         this.pageFeetRegression.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.pageFeetRegression.Name = "pageFeetRegression";
         this.pageFeetRegression.Size = new System.Drawing.Size(1157, 742);
         this.pageFeetRegression.TabIndex = 0;
         // 
         // pageObjectDetection
         // 
         this.pageObjectDetection.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pageObjectDetection.Location = new System.Drawing.Point(0, 0);
         this.pageObjectDetection.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.pageObjectDetection.Name = "pageObjectDetection";
         this.pageObjectDetection.Size = new System.Drawing.Size(1157, 742);
         this.pageObjectDetection.TabIndex = 0;
         // 
         // MainForm
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.ClientSize = new System.Drawing.Size(1165, 770);
         this.Controls.Add(this.tabControlMain);
         this.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.Name = "MainForm";
         this.Text = "MainForm";
         this.tabControlMain.ResumeLayout(false);
         this.tabPageTextMeaning.ResumeLayout(false);
         this.tabPageObjectDetection.ResumeLayout(false);
         this.tabPageImageClassification.ResumeLayout(false);
         this.tabPageFeetRegression.ResumeLayout(false);
         this.ResumeLayout(false);

      }

      #endregion
      private System.Windows.Forms.TabControl tabControlMain;
      private System.Windows.Forms.TabPage tabPageFeetRegression;
      private System.Windows.Forms.TabPage tabPageImageClassification;
      private PageImageClassification pageImageClassification;
      private System.Windows.Forms.TabPage tabPageTextMeaning;
      private PageTextMeaning pageTextMeaning;
      private PageFeetRegression pageFeetRegression;
      private System.Windows.Forms.TabPage tabPageObjectDetection;
      private PageObjectDetection pageObjectDetection;
   }
}

