
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
         this.tabPageImageClassification = new System.Windows.Forms.TabPage();
         this.pageImageClassification = new MachineLearningStudio.PageImageClassification();
         this.tabPageFeetRegression = new System.Windows.Forms.TabPage();
         this.tabPageFeetKMeans = new System.Windows.Forms.TabPage();
         this.pageFeetKMeans = new MachineLearningStudio.PageFeetKMeans();
         this.tabPageIrisKMeans = new System.Windows.Forms.TabPage();
         this.pageIrisKMeans = new MachineLearningStudio.PageIrisKMeans();
         this.pageFeetRegression = new MachineLearningStudio.PageFeetRegression();
         this.tabControlMain.SuspendLayout();
         this.tabPageTextMeaning.SuspendLayout();
         this.tabPageImageClassification.SuspendLayout();
         this.tabPageFeetRegression.SuspendLayout();
         this.tabPageFeetKMeans.SuspendLayout();
         this.tabPageIrisKMeans.SuspendLayout();
         this.SuspendLayout();
         // 
         // tabControlMain
         // 
         this.tabControlMain.Controls.Add(this.tabPageTextMeaning);
         this.tabControlMain.Controls.Add(this.tabPageImageClassification);
         this.tabControlMain.Controls.Add(this.tabPageFeetRegression);
         this.tabControlMain.Controls.Add(this.tabPageFeetKMeans);
         this.tabControlMain.Controls.Add(this.tabPageIrisKMeans);
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
         this.tabPageFeetRegression.Text = "Feed Regression";
         this.tabPageFeetRegression.UseVisualStyleBackColor = true;
         // 
         // tabPageFeetKMeans
         // 
         this.tabPageFeetKMeans.Controls.Add(this.pageFeetKMeans);
         this.tabPageFeetKMeans.Location = new System.Drawing.Point(4, 24);
         this.tabPageFeetKMeans.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.tabPageFeetKMeans.Name = "tabPageFeetKMeans";
         this.tabPageFeetKMeans.Padding = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.tabPageFeetKMeans.Size = new System.Drawing.Size(1157, 742);
         this.tabPageFeetKMeans.TabIndex = 0;
         this.tabPageFeetKMeans.Text = "Feet K-Means";
         this.tabPageFeetKMeans.UseVisualStyleBackColor = true;
         // 
         // pageFeetKMeans
         // 
         this.pageFeetKMeans.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pageFeetKMeans.Location = new System.Drawing.Point(4, 3);
         this.pageFeetKMeans.Margin = new System.Windows.Forms.Padding(5, 3, 5, 3);
         this.pageFeetKMeans.Name = "pageFeetKMeans";
         this.pageFeetKMeans.Size = new System.Drawing.Size(1149, 736);
         this.pageFeetKMeans.TabIndex = 0;
         // 
         // tabPageIrisKMeans
         // 
         this.tabPageIrisKMeans.Controls.Add(this.pageIrisKMeans);
         this.tabPageIrisKMeans.Location = new System.Drawing.Point(4, 24);
         this.tabPageIrisKMeans.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.tabPageIrisKMeans.Name = "tabPageIrisKMeans";
         this.tabPageIrisKMeans.Padding = new System.Windows.Forms.Padding(4, 3, 4, 3);
         this.tabPageIrisKMeans.Size = new System.Drawing.Size(1157, 742);
         this.tabPageIrisKMeans.TabIndex = 1;
         this.tabPageIrisKMeans.Text = "Iris K-Means";
         this.tabPageIrisKMeans.UseVisualStyleBackColor = true;
         // 
         // pageIrisKMeans
         // 
         this.pageIrisKMeans.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pageIrisKMeans.Location = new System.Drawing.Point(4, 3);
         this.pageIrisKMeans.Margin = new System.Windows.Forms.Padding(5, 3, 5, 3);
         this.pageIrisKMeans.Name = "pageIrisKMeans";
         this.pageIrisKMeans.Size = new System.Drawing.Size(1149, 736);
         this.pageIrisKMeans.TabIndex = 0;
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
         this.tabPageImageClassification.ResumeLayout(false);
         this.tabPageFeetRegression.ResumeLayout(false);
         this.tabPageFeetKMeans.ResumeLayout(false);
         this.tabPageIrisKMeans.ResumeLayout(false);
         this.ResumeLayout(false);

      }

      #endregion
      private System.Windows.Forms.TabControl tabControlMain;
      private System.Windows.Forms.TabPage tabPageFeetKMeans;
      private System.Windows.Forms.TabPage tabPageIrisKMeans;
      private PageFeetKMeans pageFeetKMeans;
      private PageIrisKMeans pageIrisKMeans;
      private System.Windows.Forms.TabPage tabPageFeetRegression;
      private System.Windows.Forms.TabPage tabPageImageClassification;
      private PageImageClassification pageImageClassification;
      private System.Windows.Forms.TabPage tabPageTextMeaning;
      private PageTextMeaning pageTextMeaning;
      private PageFeetRegression pageFeetRegression;
   }
}

