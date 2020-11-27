
namespace TestChoice
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
         this.labelPrediction = new System.Windows.Forms.Label();
         this.tabControlMain = new System.Windows.Forms.TabControl();
         this.tabPageFeetKMeans = new System.Windows.Forms.TabPage();
         this.tabPageIrisKMeans = new System.Windows.Forms.TabPage();
         this.pageFeetKMeans1 = new TestChoice.PageFeetKMeans();
         this.pageIrisKMeans1 = new TestChoice.PageIrisKMeans();
         this.tabControlMain.SuspendLayout();
         this.tabPageFeetKMeans.SuspendLayout();
         this.tabPageIrisKMeans.SuspendLayout();
         this.SuspendLayout();
         // 
         // labelPrediction
         // 
         this.labelPrediction.AutoSize = true;
         this.labelPrediction.Location = new System.Drawing.Point(373, 18);
         this.labelPrediction.Name = "labelPrediction";
         this.labelPrediction.Size = new System.Drawing.Size(16, 13);
         this.labelPrediction.TabIndex = 3;
         this.labelPrediction.Text = "...";
         // 
         // tabControlMain
         // 
         this.tabControlMain.Controls.Add(this.tabPageFeetKMeans);
         this.tabControlMain.Controls.Add(this.tabPageIrisKMeans);
         this.tabControlMain.Dock = System.Windows.Forms.DockStyle.Fill;
         this.tabControlMain.Location = new System.Drawing.Point(0, 0);
         this.tabControlMain.Name = "tabControlMain";
         this.tabControlMain.SelectedIndex = 0;
         this.tabControlMain.Size = new System.Drawing.Size(1049, 737);
         this.tabControlMain.TabIndex = 5;
         // 
         // tabPageFeetKMeans
         // 
         this.tabPageFeetKMeans.Controls.Add(this.pageFeetKMeans1);
         this.tabPageFeetKMeans.Location = new System.Drawing.Point(4, 22);
         this.tabPageFeetKMeans.Name = "tabPageFeetKMeans";
         this.tabPageFeetKMeans.Padding = new System.Windows.Forms.Padding(3);
         this.tabPageFeetKMeans.Size = new System.Drawing.Size(1041, 711);
         this.tabPageFeetKMeans.TabIndex = 0;
         this.tabPageFeetKMeans.Text = "Feet K-Means";
         this.tabPageFeetKMeans.UseVisualStyleBackColor = true;
         // 
         // tabPageIrisKMeans
         // 
         this.tabPageIrisKMeans.Controls.Add(this.pageIrisKMeans1);
         this.tabPageIrisKMeans.Location = new System.Drawing.Point(4, 22);
         this.tabPageIrisKMeans.Name = "tabPageIrisKMeans";
         this.tabPageIrisKMeans.Padding = new System.Windows.Forms.Padding(3);
         this.tabPageIrisKMeans.Size = new System.Drawing.Size(1041, 711);
         this.tabPageIrisKMeans.TabIndex = 1;
         this.tabPageIrisKMeans.Text = "Iris K-Means";
         this.tabPageIrisKMeans.UseVisualStyleBackColor = true;
         // 
         // pageFeetKMeans1
         // 
         this.pageFeetKMeans1.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pageFeetKMeans1.Location = new System.Drawing.Point(3, 3);
         this.pageFeetKMeans1.Name = "pageFeetKMeans1";
         this.pageFeetKMeans1.Size = new System.Drawing.Size(1035, 705);
         this.pageFeetKMeans1.TabIndex = 0;
         // 
         // pageIrisKMeans1
         // 
         this.pageIrisKMeans1.Dock = System.Windows.Forms.DockStyle.Fill;
         this.pageIrisKMeans1.Location = new System.Drawing.Point(3, 3);
         this.pageIrisKMeans1.Name = "pageIrisKMeans1";
         this.pageIrisKMeans1.Size = new System.Drawing.Size(1035, 705);
         this.pageIrisKMeans1.TabIndex = 0;
         // 
         // MainForm
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.ClientSize = new System.Drawing.Size(1049, 737);
         this.Controls.Add(this.tabControlMain);
         this.Controls.Add(this.labelPrediction);
         this.Name = "MainForm";
         this.Text = "Form1";
         this.tabControlMain.ResumeLayout(false);
         this.tabPageFeetKMeans.ResumeLayout(false);
         this.tabPageIrisKMeans.ResumeLayout(false);
         this.ResumeLayout(false);
         this.PerformLayout();

      }

      #endregion
      private System.Windows.Forms.Label labelPrediction;
      private System.Windows.Forms.TabControl tabControlMain;
      private System.Windows.Forms.TabPage tabPageFeetKMeans;
      private System.Windows.Forms.TabPage tabPageIrisKMeans;
      private PageFeetKMeans pageFeetKMeans1;
      private PageIrisKMeans pageIrisKMeans1;
   }
}

