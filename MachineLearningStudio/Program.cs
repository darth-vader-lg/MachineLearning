using System;
using System.Windows.Forms;

namespace TestChoice
{
   /// <summary>
   /// Programma
   /// </summary>
   internal static class Program
   {
      #region Methods
      /// <summary>
      /// Punto di ingresso principale dell'applicazione.
      /// </summary>
      [STAThread]
      private static void Main()
      {
         Application.EnableVisualStyles();
         Application.SetCompatibleTextRenderingDefault(false);
         Application.Run(new MainForm());
      }
      #endregion
   }
}
