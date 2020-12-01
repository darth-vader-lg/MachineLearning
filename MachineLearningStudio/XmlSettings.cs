using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Runtime.Serialization;
using System.Text;
using System.Xml;
using System.Xml.Serialization;
using System.Xml.XPath;

namespace MachineLearningStudio
{
   /// <summary>
   /// base per impostazioni in formato XML
   /// </summary>
   [Serializable]
   [XmlType("XmlSettings")]
   public partial class XmlSettings
   {
      #region Color
      [Serializable]
      public struct Color
      {
         #region Fields
         /// <summary>
         /// Colore
         /// </summary>
         private System.Drawing.Color color;
         /// <summary>
         /// Colore
         /// </summary>
         private string name;
         #endregion
         #region Properties
         /// <summary>
         /// Alfa
         /// </summary>
         public string Name { get { return name; } }
         /// <summary>
         /// Alfa
         /// </summary>
         public int A { get { return color.A; } }
         /// <summary>
         /// Blu
         /// </summary>
         public int B { get { return color.B; } }
         /// <summary>
         /// Verde
         /// </summary>
         public int G { get { return color.G; } }
         /// <summary>
         /// Rosso
         /// </summary>
         public int R { get { return color.R; } }
         /// <summary>
         /// Colore in formato HTML
         /// </summary>
         [XmlAttribute]
         public string ColorHtml
         {
            get
            {
               if (color != System.Drawing.Color.Empty && color.A != 0xff) {
                  var namedColor = System.Drawing.Color.FromName(name);
                  return (namedColor != System.Drawing.Color.Empty ? name : ColorTranslator.ToHtml(color)) + ";" + color.A.ToString("X2");
               }
               else
                  return ColorTranslator.ToHtml(color);
            }
            set
            {
               // Splitta la stringa in tokens separati da ;
               var tokens = value.Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
               if (tokens != null && tokens.Length > 0) {
                  // Verifica se vecchio formato di salvataggio con colore html
                  if (tokens[0].StartsWith("#") && tokens.Length < 2) {
                     color = System.Drawing.Color.FromArgb(tokens.Length > 1 ? Convert.ToInt32(tokens[1]) : 0xff, ColorTranslator.FromHtml(tokens[0]));
                     name = color.Name;
                  }
                  // Nuovo formato customizzato per evitare il problema della ColorTranslator che si mangia il canale alpha
                  else {
                     // Tenta la conversione da nome senza canale alpha
                     color = ColorTranslator.FromHtml(tokens[0]);
                     name = color.Name;
                     if (tokens.Length > 1)
                        color = System.Drawing.Color.FromArgb(Convert.ToInt32(tokens[1], 16), color);
                  }
               }
               else
                  color = System.Drawing.Color.Empty;
            }
         }
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="color">Colore di tipo System.Drawing</param>
         public Color(System.Drawing.Color color)
         {
            name = color.Name;
            this.color = color;
         }
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="alpha">Componente alpha</param>
         /// <param name="baseColor">Colore di base di tipo System.Drawing</param>
         public Color(int alpha, System.Drawing.Color baseColor)
         {
            name = baseColor.Name;
            color = System.Drawing.Color.FromArgb(alpha, baseColor);
         }
         /// <summary>
         /// Conversione a System.Drawing.Color
         /// </summary>
         /// <param name="color">Colore</param>
         public static implicit operator System.Drawing.Color(Color color)
         {
            return color.color;
         }
         /// <summary>
         /// Conversione da System.Drawing.Color
         /// </summary>
         /// <param name="color">Colore</param>
         public static implicit operator Color(System.Drawing.Color color)
         {
            return new Color(color);
         }
         /// <summary>
         /// Override della funzione di rappresentazione in formato stringa
         /// </summary>
         /// <returns></returns>
         public override string ToString()
         {
            return string.Format("{0} [A={1:X2}, R={2:X2}, G={3:X2}, B={4:X2}", name + (color.A >= 0xff ? "" : string.Format("/{0}", color.A)), color.A, color.R, color.G, color.B);
         }
         #endregion
      }
      #endregion
      #region Fields
      /// <summary>
      /// Elenco di tipi extra conosciuti
      /// </summary>
      [NonSerialized]
      Type[] extraTypes;
      /// <summary>
      /// Elenco di tipi extra conosciuti filtrati fra quelli che possono essere serializzati in xml
      /// </summary>
      [NonSerialized]
      Type[] extraTypesFiltered;
      /// <summary>
      /// Path del file
      /// </summary>
      string path;
      #endregion
      #region Properties
      /// <summary>
      /// Formattazione custom dell'xml per non perdere i cr/lf scritti in forma &#13; &#10;
      /// </summary>
      [XmlIgnore]
      public bool CustomStringFormatter { get; set; }
      /// <summary>
      /// Elenco di tipi extra conosciuti
      /// </summary>
      [XmlIgnore]
      protected Type[] ExtraTypes
      {
         get { return extraTypes; }
         set { extraTypesFiltered = value != extraTypes ? null : extraTypesFiltered; extraTypes = value; }
      }
      /// <summary>
      /// Elenco di tipi extra conosciuti
      /// </summary>
      [XmlIgnore]
      private Type[] ExtraTypesFiltered
      {
         get
         {
            if (extraTypesFiltered == null) {
               if (extraTypes != null && extraTypes.Length > 0) {
                  // Filtra l'elenco dei tipi serializzabili dall'xml
                  var list = new List<Type>(extraTypes);
                  list.RemoveAll(type => type.GetCustomAttributes(typeof(SerializableAttribute), true).Length == 0);
                  while (list.Count > 0) {
                     try {
                        // Crea un serializzatore per provare sela lista dei tipi è valida
                        var serializer = new XmlSerializer(GetType(), list.ToArray() ?? new Type[0]);
                        break;
                     }
                     catch (Exception exc) {
                        // Rimuove il tipo contenuto nel messaggio di eccezione dalla lista dei tipi conosciuti
                        if (list.RemoveAll(type => exc.Message.Contains(type.FullName.Replace('+', '.'))) <= 0)
                           break;
                     }
                  }
                  extraTypesFiltered = list.ToArray();
               }
               else
                  extraTypesFiltered = new Type[0];
            }
            return extraTypesFiltered;
         }
      }
      /// <summary>
      /// Path attuale dei settings
      /// </summary>
      public string FilePath { get { return path; } }
      /// <summary>
      /// Flag di caricamento avvenuto correttamente
      /// </summary>
      [XmlIgnore]
      public bool Loaded { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Caricamento da file
      /// </summary>
      /// <param name="path">Path del file da caricare</param>
      /// <returns>Restituisce l'oggetto contenente la configurazione xml</returns>
      public static T Load<T>(string path) where T : XmlSettings, new()
      {
         var result = new T();
         result.OnLoading();
         result.Load(path);
         return result;
      }
      /// <summary>
      /// Caricamento da file
      /// </summary>
      /// <param name="path">Path del file da caricare</param>
      public void Load(string path)
      {
         this.path = path;
         var customStringFormatter = CustomStringFormatter;
         using (XmlReader reader = XmlReader.Create(path, new XmlReaderSettings() { })) {
            var settings = new XmlSerializer(GetType(), ExtraTypesFiltered ?? new Type[0]).Deserialize(reader);
            var members = FormatterServices.GetSerializableMembers(GetType());
            FormatterServices.PopulateObjectMembers(this, members, FormatterServices.GetObjectData(settings, members));
            CustomStringFormatter = customStringFormatter;
            OnLoaded();
            Loaded = true;
         }
      }
      /// <summary>
      /// Funziona chiamata al termine del caricamento
      /// </summary>
      protected virtual void OnLoaded()
      {
      }
      /// <summary>
      /// Funziona chiamata prima del caricamento
      /// </summary>
      protected virtual void OnLoading()
      {
      }
      /// <summary>
      /// Funziona chiamata al termine del salvataggio
      /// </summary>
      protected virtual void OnSaved()
      {
      }
      /// <summary>
      /// Funziona chiamata prima del salvataggio
      /// </summary>
      protected virtual void OnSaving()
      {
      }
      /// <summary>
      /// Salva in un file
      /// </summary>
      /// <param name="path">Path del file (o null per path attuale)</param>
      public void Save(string path = null)
      {
         // Utilizza path di caricamento se path di salvataggio non definito
         path = string.IsNullOrWhiteSpace(path) ? this.path : path;
         // Chiama la funzione di inizio salvataggio
         OnSaving();
         // Serializza
         Serialize(path);
         // Aggiorna il path
         this.path = path;
         // Chiama la funzione post serializzazione
         OnSaved();
      }
      /// <summary>
      /// Salva una copia del file
      /// </summary>
      /// <param name="path">Path della copia</param>
      public void SaveCopy(string path)
      {
         // Chiama la funzione di inizio salvataggio
         OnSaving();
         // Serializza
         Serialize(path);
         // Chiama la funzione post serializzazione
         OnSaved();
      }
      /// <summary>
      /// Serializza in un file
      /// </summary>
      /// <param name="path">Path del file</param>
      private void Serialize(string path)
      {
         // Ottiene path completo del file
         path = Path.GetFullPath(path);
         // Path temporaneo di scrittura
         var tmpPath = Path.GetTempFileName();
         try {
            // Directory del file
            var dir = Path.GetDirectoryName(path);
            // Crea la directory se non esiste
            if (!Directory.Exists(Path.GetDirectoryName(path)))
               Directory.CreateDirectory(dir);
            // Serializza l'oggetto
            using (var writer = new Writer(tmpPath, CustomStringFormatter))
               new XmlSerializer(GetType(), ExtraTypesFiltered ?? new Type[0]).Serialize(writer, this);
            // Copia il file nella destinazione finale
            File.Copy(tmpPath, path, true);
         }
         finally {
            if (File.Exists(tmpPath))
               File.Delete(tmpPath);
         }
      }
      #endregion
   }

   /// <summary>
   /// Implementazione custom della XmlWriter
   /// </summary>
   partial class XmlSettings
   {
      #region Writer
      class Writer : XmlWriter
      {
         #region Fields
         /// <summary>
         /// Formattazione custom delle stringhe
         /// </summary>
         private readonly bool customStringFormatter;
         /// <summary>
         /// Writer originale
         /// </summary>
         private readonly XmlWriter writer;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="path">Path di salvataggio del file</param>
         /// <param name="customStringFormatter">Formattazione custom dell'xml per non perdere i cr/lf scrittti in forma &#13; &#10;</param>
         public Writer(string path, bool customStringFormatter)
         {
            // Crea i settings per l'xml writer
            this.customStringFormatter = customStringFormatter;
            var settings = new XmlWriterSettings() { Indent = true };
            writer = Create(path, settings);
         }
         public override void Close()
         {
            writer.Close();
         }
         protected override void Dispose(bool disposing)
         {
            base.Dispose(disposing);
            if (disposing)
               (writer as IDisposable).Dispose();
         }
         public override void Flush()
         {
            writer.Flush();
         }
         public override string LookupPrefix(string ns)
         {
            return writer.LookupPrefix(ns);
         }
         public override XmlWriterSettings Settings
         {
            get
            {
               return writer.Settings;
            }
         }
         public override string ToString()
         {
            return writer.ToString();
         }
         public override void WriteAttributes(XmlReader reader, bool defattr)
         {
            writer.WriteAttributes(reader, defattr);
         }
         public override void WriteBase64(byte[] buffer, int index, int count)
         {
            writer.WriteBase64(buffer, index, count);
         }
         public override void WriteBinHex(byte[] buffer, int index, int count)
         {
            writer.WriteBinHex(buffer, index, count);
         }
         public override void WriteCData(string text)
         {
            writer.WriteCData(text);
         }
         public override void WriteCharEntity(char ch)
         {
            writer.WriteCharEntity(ch);
         }
         public override void WriteChars(char[] buffer, int index, int count)
         {
            writer.WriteChars(buffer, index, count);
         }
         public override void WriteComment(string text)
         {
            writer.WriteComment(text);
         }
         public override void WriteDocType(string name, string pubid, string sysid, string subset)
         {
            writer.WriteDocType(name, pubid, sysid, subset);
         }
         public override void WriteEndAttribute()
         {
            writer.WriteEndAttribute();
         }
         public override void WriteEndDocument()
         {
            writer.WriteEndDocument();
         }
         public override void WriteEndElement()
         {
            writer.WriteEndElement();
         }
         public override void WriteEntityRef(string name)
         {
            writer.WriteEntityRef(name);
         }
         public override void WriteFullEndElement()
         {
            writer.WriteFullEndElement();
         }
         public override void WriteName(string name)
         {
            writer.WriteName(name);
         }
         public override void WriteNmToken(string name)
         {
            writer.WriteNmToken(name);
         }
         public override void WriteNode(XPathNavigator navigator, bool defattr)
         {
            writer.WriteNode(navigator, defattr);
         }
         public override void WriteNode(XmlReader reader, bool defattr)
         {
            writer.WriteNode(reader, defattr);
         }
         public override void WriteProcessingInstruction(string name, string text)
         {
            writer.WriteProcessingInstruction(name, text);
         }
         public override void WriteQualifiedName(string localName, string ns)
         {
            writer.WriteQualifiedName(localName, ns);
         }
         public override void WriteRaw(char[] buffer, int index, int count)
         {
            writer.WriteRaw(buffer, index, count);
         }
         public override void WriteRaw(string data)
         {
            writer.WriteRaw(data);
         }
         public override void WriteStartAttribute(string prefix, string localName, string ns)
         {
            writer.WriteStartAttribute(prefix, localName, ns);
         }
         public override void WriteStartDocument()
         {
            writer.WriteStartDocument();
         }
         public override void WriteStartDocument(bool standalone)
         {
            writer.WriteStartDocument(standalone);
         }
         public override void WriteStartElement(string prefix, string localName, string ns)
         {
            writer.WriteStartElement(prefix, localName, ns);
         }
         public override WriteState WriteState
         {
            get
            {
               return writer.WriteState;
            }
         }
         public override void WriteString(string text)
         {
            if (customStringFormatter) {
               var xmlText = new StringBuilder(text.Length);
               for (var i = 0; i < text.Length; i++) {
                  switch (text[i]) {
                     case '&': xmlText.Append("&amp;"); break;
                     case '<': xmlText.Append("&lt;"); break;
                     case '>': xmlText.Append("&gt;"); break;
                     case '"': xmlText.Append("&quot;"); break;
                     case '\t':
                     case '\r':
                     case '\n': xmlText.AppendFormat("&#{0};", (int)text[i]); break;
                     default: xmlText.Append(text[i]); break;
                  }
               }
               writer.WriteRaw(xmlText.ToString());
            }
            else
               writer.WriteString(text);
         }
         public override void WriteSurrogateCharEntity(char lowChar, char highChar)
         {
            writer.WriteSurrogateCharEntity(lowChar, highChar);
         }
         public override void WriteValue(bool value)
         {
            writer.WriteValue(value);
         }
         public override void WriteValue(DateTime value)
         {
            writer.WriteValue(value);
         }
         public override void WriteValue(decimal value)
         {
            writer.WriteValue(value);
         }
         public override void WriteValue(double value)
         {
            writer.WriteValue(value);
         }
         public override void WriteValue(float value)
         {
            writer.WriteValue(value);
         }
         public override void WriteValue(int value)
         {
            writer.WriteValue(value);
         }
         public override void WriteValue(long value)
         {
            writer.WriteValue(value);
         }
         public override void WriteValue(object value)
         {
            writer.WriteValue(value);
         }
         public override void WriteValue(string value)
         {
            writer.WriteValue(value);
         }
         public override void WriteWhitespace(string ws)
         {
            writer.WriteWhitespace(ws);
         }
         public override string XmlLang
         {
            get
            {
               return writer.XmlLang;
            }
         }
         public override XmlSpace XmlSpace
         {
            get
            {
               return writer.XmlSpace;
            }
         }
         #endregion
      }
      #endregion
   }
}