namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per i providers di opzioni di testo
   /// </summary>
   public interface ITextDataOptionsProvider
   {
      #region Properties
      /// <summary>
      /// Opzioni di caricamento dati testuali
      /// </summary>
      TextDataOptions TextDataOptions { get; }
      #endregion
   }
}
