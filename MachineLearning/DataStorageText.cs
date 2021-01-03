using Microsoft.ML;
using System;

namespace MachineLearning
{
   /// <summary>
   /// Classe base per lo storage di dati di tipo file di testo
   /// </summary>
   [Serializable]
   public abstract class DataStorageText : DataStorageBase, IDataStorage
   {
      #region Properties
      /// <summary>
      /// Specifica se salvare il commento con lo schema all'inizio del file
      /// </summary>
      public bool SaveSchema { get; private set; }
      /// <summary>
      /// Specifica se mantenere le colonne nascoste nel set di dati
      /// </summary>
      public bool KeepHidden { get; private set; }
      /// <summary>
      /// Specifica se salvare le colonne in formato denso anche se sono vettori sparsi
      /// </summary>
      public bool ForceDense { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      public virtual IDataView LoadData(object context) => LoadTextData(context);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public virtual void SaveData(object context, IDataView data) { }
      #endregion
   }
}
